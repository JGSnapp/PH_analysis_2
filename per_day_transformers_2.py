# per_day_transformers_2.py
# Групповой ранкинг по дням (Product Hunt), per-day Spearman, SBERT fine-tune,
# антислип и сохранение артефактов (совместимо с разными версиями sentence-transformers/LightGBM)

import os
import sys
import json
import math
import argparse
import logging
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
import joblib

import lightgbm as lgb
from lightgbm import LGBMRegressor, LGBMRanker
from xgboost import XGBRegressor

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# -------- antisleep (безопасный фоллбэк) --------
try:
    from antisleep import keep_awake
except Exception:
    from contextlib import contextmanager
    @contextmanager
    def keep_awake(_msg="keep-awake", logger=None):
        if logger: logger.info("keep_awake: fallback (no-op)")
        yield

# =========================
# Утилиты
# =========================
def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

def join_text(row):
    return f"{row.get('tagline','')} {row.get('description','')} {row.get('topics_str','')}".strip()

def spearman_global(y_true, y_pred):
    return float(spearmanr(y_true, y_pred).correlation)

def spearman_per_day(df_part: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    d = df_part[["ph_day"]].copy()
    d["y"] = y_true
    d["p"] = y_pred
    vals = []
    for _, g in d.groupby("ph_day"):
        if len(g) >= 3:
            vals.append(g["y"].rank().corr(g["p"].rank(), method="spearman"))
    return float(np.nanmean(vals)) if len(vals) else np.nan

def order_by_day(df_part: pd.DataFrame):
    idx_blocks, groups, day_list = [], [], []
    for day, g in df_part.groupby("ph_day", sort=True):
        idx_blocks.append(g.index.to_list())
        groups.append(len(g))
        day_list.append(day)
    ordered_idx = sum(idx_blocks, [])
    return ordered_idx, groups, day_list

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_json(obj, path: Path):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

# Дискретизация перцентиля в целочисленные уровни релевантности (0..levels-1)
def to_relevance_levels(percentiles: np.ndarray, levels: int = 5) -> np.ndarray:
    pct = np.clip(percentiles, 0.0, 1.0)
    rel = np.floor(np.minimum(pct * levels, levels - 1e-9)).astype(int)
    return rel  # dtype=int

# =========================
# Основной сценарий
# =========================
def main():
    parser = argparse.ArgumentParser(description="Per-day ranking with SBERT + LGBMRanker (antisleep + save artifacts)")
    parser.add_argument("--input", type=str, default="output.xlsx", help="Путь к Excel с данными")
    parser.add_argument("--outdir", type=str, default=None, help="Директория для артефактов (по умолчанию ./artifacts/per_day_{ts})")
    parser.add_argument("--sbert", type=str, default="all-MiniLM-L6-v2", help="Путь к модели SBERT или имя хаба")
    parser.add_argument("--epochs", type=int, default=4, help="SBERT: число эпох fine-tune")
    parser.add_argument("--batch", type=int, default=64, help="SBERT: batch size (в MNRL больше — лучше)")
    parser.add_argument("--lr", type=float, default=2e-5, help="SBERT: learning rate")
    parser.add_argument("--scale", type=float, default=20.0, help="SBERT: temperature (scale) для MultipleNegativesRankingLoss")
    parser.add_argument("--topk", type=int, default=12, help="Формирование пар: TOP_K")
    parser.add_argument("--botk", type=int, default=60, help="Формирование пар: BOT_K")
    parser.add_argument("--min_delta", type=float, default=0.30, help="Мин. разница перцентилей для пары")
    parser.add_argument("--test_frac_days", type=float, default=0.2, help="Доля последних дней в тесте")
    parser.add_argument("--rel_levels", type=int, default=20, help="Число уровней релевантности для LambdaRank (int labels)")
    parser.add_argument("--gain", type=str, default="linear", choices=["linear","exp"],
                        help="Схема label_gain для LambdaRank: linear (0..L-1) или exp (2^i-1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Логи
    log = logging.getLogger("per_day")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    set_seeds(args.seed)

    # Куда сохраняем артефакты
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir) if args.outdir else Path("artifacts") / f"per_day_{ts}"
    ensure_dir(outdir); ensure_dir(outdir / "models")

    meta = {
        "timestamp": ts,
        "input": args.input,
        "sbert_base": args.sbert,
        "epochs": args.epochs,
        "batch": args.batch,
        "lr": args.lr,
        "scale": args.scale,
        "pairs": {"TOP_K": args.topk, "BOT_K": args.botk, "MIN_DELTA": args.min_delta},
        "test_frac_days": args.test_frac_days,
        "rel_levels": args.rel_levels,
        "gain": args.gain,
        "seed": args.seed,
    }
    save_json(meta, outdir / "metadata.json")

    with keep_awake("Per-day: SBERT fine-tune + feature build + ranker training + saving", logger=log):
        # ---------- 1) Загрузка и базовая чистка ----------
        log.info(f"Загружаю данные из {args.input} ...")
        df = pd.read_excel(args.input).copy()

        df["votesCount"] = pd.to_numeric(df.get("votesCount"), errors="coerce").fillna(0).astype(int)
        df["commentsCount"] = pd.to_numeric(df.get("commentsCount"), errors="coerce").fillna(0).astype(int)
        df["createdAt"] = pd.to_datetime(df.get("createdAt"), errors="coerce")

        df["ph_day"] = df["createdAt"].dt.date.astype("string")
        df.loc[df["createdAt"].isna(), "ph_day"] = "unknown"

        df["day_N"] = df.groupby("ph_day")["id"].transform("size").astype(int)
        df["day_votes_min"] = df.groupby("ph_day")["votesCount"].transform("min").astype(int)
        df["day_votes_max"] = df.groupby("ph_day")["votesCount"].transform("max").astype(int)
        df["day_rank"] = df.groupby("ph_day")["votesCount"].rank(method="average", ascending=False)
        df["day_percentile"] = (df["day_N"] - df["day_rank"] + 0.5) / df["day_N"].clip(lower=1)

        df_model = df.loc[df["ph_day"] != "unknown"].copy()

        # ---------- 2) Фичи ----------
        df_model["tagline_len"] = df_model["tagline"].fillna("").str.len()
        df_model["desc_len"]    = df_model["description"].fillna("").str.len()
        for col in ["img_count", "video_count", "topics_count"]:
            if col not in df_model.columns:
                df_model[col] = 0

        df_model["dow"] = df_model["createdAt"].dt.dayofweek
        df_model["dow_sin"] = np.sin(2 * np.pi * df_model["dow"] / 7)
        df_model["dow_cos"] = np.cos(2 * np.pi * df_model["dow"] / 7)

        if "topics_str" not in df_model.columns:
            df_model["topics_str"] = ""
        df_model["topics_str"] = df_model["topics_str"].fillna("")
        df_model["text_all"] = df_model.apply(join_text, axis=1)

        num_cols = ["img_count", "video_count", "topics_count", "tagline_len", "desc_len",
                    "dow_sin", "dow_cos", "day_N"]
        (outdir / "models" / "num_cols.json").write_text(json.dumps(num_cols), encoding="utf-8")

        # ---------- 3) Time-based split ----------
        all_days_sorted = (
            pd.to_datetime(df_model["ph_day"]).sort_values().drop_duplicates().astype("string").tolist()
        )
        split_idx = max(1, int(math.ceil(len(all_days_sorted) * (1 - args.test_frac_days))))
        train_days = set(all_days_sorted[:split_idx])
        test_days  = set(all_days_sorted[split_idx:])

        train_df = df_model[df_model["ph_day"].isin(train_days)].copy()
        test_df  = df_model[df_model["ph_day"].isin(test_days)].copy()

        y_tr = train_df["day_percentile"].to_numpy(dtype=float)
        y_te = test_df["day_percentile"].to_numpy(dtype=float)

        log.info(f"Train days={len(train_days)} (rows={len(train_df)}), Test days={len(test_days)} (rows={len(test_df)})")
        if len(test_df) < 3:
            log.warning("Слишком мало данных в тесте для стабильной оценки Spearman per-day.")

        # ---------- 4) SBERT: контрастные пары и дообучение ----------
        # Сохраняем базовую модель сразу (даже если что-то упадёт)
        sbert = SentenceTransformer(args.sbert)
        sbert_dir = ensure_dir(outdir / "models" / "sbert_model")
        sbert.save(str(sbert_dir))
        log.info(f"SBERT (текущая версия модели) сохранён в: {sbert_dir}")

        TOP_K, BOT_K, MIN_DELTA = args.topk, args.botk, args.min_delta
        train_examples = []
        if args.epochs > 0 and (TOP_K > 0 or BOT_K > 0):
            for _, g in train_df.groupby("ph_day", sort=False):
                if len(g) < 3: continue
                g = g.sort_values("day_percentile", ascending=False)
                gt = g.head(TOP_K)
                gb = g.tail(BOT_K)

                for _, a in gt.iterrows():
                    for _, b in gb.iterrows():
                        if (a["day_percentile"] - b["day_percentile"]) < MIN_DELTA:
                            continue
                        train_examples.append(InputExample(texts=[a["text_all"], b["text_all"]]))
                gt_list = gt["text_all"].tolist()
                for i in range(len(gt_list) - 1):
                    train_examples.append(InputExample(texts=[gt_list[i], gt_list[i+1]]))
                gb_list = gb["text_all"].tolist()
                for i in range(len(gb_list) - 1):
                    train_examples.append(InputExample(texts=[gb_list[i], gb_list[i+1]]))

        if len(train_examples) > 0:
            log.info(f"SBERT: обучающих пар = {len(train_examples)}")
            train_loader = DataLoader(train_examples, shuffle=True, batch_size=args.batch)
            train_loss = losses.MultipleNegativesRankingLoss(sbert, scale=float(args.scale))

            try:
                sbert.fit(
                    train_objectives=[(train_loader, train_loss)],
                    epochs=int(args.epochs),
                    warmup_steps=min(1000, max(10, len(train_loader)//10)),
                    optimizer_params={'lr': float(args.lr)},
                    scheduler="WarmupCosine",   # если недоступно — отловим ниже
                    max_grad_norm=1.0,
                    show_progress_bar=True
                )
            except Exception as e1:
                if "Unknown scheduler" in str(e1):
                    log.warning("SBERT: WarmupCosine недоступен, пробую дефолтный шедулер (WarmupLinear).")
                    sbert.fit(
                        train_objectives=[(train_loader, train_loss)],
                        epochs=int(args.epochs),
                        warmup_steps=min(1000, max(10, len(train_loader)//10)),
                        optimizer_params={'lr': float(args.lr)},
                        max_grad_norm=1.0,
                        show_progress_bar=True
                    )
                else:
                    log.exception(f"SBERT: ошибка обучения: {e1}")
            # сохраняем текущую (пере)обученную модель
            sbert.save(str(sbert_dir))
            log.info(f"SBERT сохранён (после обучения/или базовый) в: {sbert_dir}")
        else:
            if args.epochs == 0:
                log.info("SBERT fine-tune отключён (--epochs 0). Использую заданную модель как есть.")
            else:
                log.warning("SBERT: пары не сформированы — пропускаю fine-tune (останется базовая/заданная модель).")

        # ---------- 5) Признаки: эмбеддинги + числа ----------
        def make_features(df_part: pd.DataFrame):
            emb = sbert.encode(df_part["text_all"].tolist(),
                               batch_size=256, normalize_embeddings=True, show_progress_bar=False)
            nums = df_part[num_cols].to_numpy(dtype=float)
            return emb, nums

        Xtr_emb, Xtr_num = make_features(train_df)
        Xte_emb, Xte_num = make_features(test_df)

        scaler = MaxAbsScaler()
        Xtr_num = scaler.fit_transform(Xtr_num)
        Xte_num = scaler.transform(Xte_num)

        Xtr = np.hstack([Xtr_emb, Xtr_num])
        Xte = np.hstack([Xte_emb, Xte_num])

        # Сохраняем числовой скейлер
        joblib.dump(scaler, outdir / "models" / "num_scaler.pkl")
        log.info(f"Сохранены скейлер и num_cols в: {outdir / 'models'}")

        # ---------- 6) Регрессоры (для сравнения) ----------
        reg_models = {
            "Ridge (alpha=3)": Ridge(alpha=3.0, fit_intercept=True),
            "SGD (MSE, l2)": SGDRegressor(loss="squared_error", penalty="l2", alpha=1e-4,
                                          max_iter=2000, random_state=args.seed),
            "SGD (Huber)": SGDRegressor(loss="huber", penalty="l2", alpha=1e-4,
                                        max_iter=2000, random_state=args.seed),
            "SGD (ElasticNet)": SGDRegressor(loss="squared_error", penalty="elasticnet", l1_ratio=0.15,
                                             alpha=1e-4, max_iter=2000, random_state=args.seed),
            "RandomForest(dense)": RandomForestRegressor(n_estimators=400, max_depth=None,
                                                         n_jobs=-1, random_state=args.seed),
            "GradBoost(dense)": GradientBoostingRegressor(random_state=args.seed),
            "XGBoost": XGBRegressor(n_estimators=800, max_depth=8, learning_rate=0.05,
                                    subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                                    random_state=args.seed, n_jobs=-1),
            "LightGBM": LGBMRegressor(objective="regression", n_estimators=1500, learning_rate=0.04,
                                      num_leaves=63, min_data_in_leaf=100, feature_fraction=0.85,
                                      bagging_fraction=0.8, bagging_freq=1, reg_lambda=2.0,
                                      force_col_wise=True, random_state=args.seed, n_jobs=-1),
        }

        rows = []
        for name, model in reg_models.items():
            model.fit(Xtr, y_tr)
            pred = model.predict(Xte)
            rows.append([
                name,
                r2_score(y_te, pred),
                mean_absolute_error(y_te, pred),
                spearman_global(y_te, pred),
                spearman_per_day(test_df, y_te, pred)
            ])
            safe_name = name.replace(" ", "_").replace("(", "_").replace(")", "_")
            joblib.dump(model, outdir / "models" / f"{safe_name}.pkl")

        # ---------- 7) Групповой ранкер (LightGBM LambdaRank с целочисленными метками) ----------
        REL_LEVELS = int(args.rel_levels)
        if args.gain == "linear":
            label_gain = list(range(REL_LEVELS))              # 0,1,2,...,L-1
        else:
            label_gain = [int(2**i - 1) for i in range(REL_LEVELS)]  # 0,1,3,7,15,...

        log.info(f"LambdaRank: rel_levels={REL_LEVELS}, gain_mode={args.gain}, label_gain[:10]={label_gain[:10]}")

        tr_idx_ordered, tr_groups, _ = order_by_day(train_df)
        te_idx_ordered, te_groups, _ = order_by_day(test_df)

        Xtr_rank = Xtr[train_df.index.get_indexer(tr_idx_ordered)]
        Xte_rank = Xte[test_df.index.get_indexer(te_idx_ordered)]

        # перцентили -> уровни релевантности (int)
        ytr_rel_all = to_relevance_levels(y_tr, levels=REL_LEVELS)
        yte_rel_all = to_relevance_levels(y_te, levels=REL_LEVELS)

        ytr_rank = ytr_rel_all[train_df.index.get_indexer(tr_idx_ordered)]
        yte_rank = yte_rel_all[test_df.index.get_indexer(te_idx_ordered)]

        ranker_params = dict(
            objective="lambdarank",
            metric="ndcg",
            eval_at=[5, 10],
            learning_rate=0.05,
            num_leaves=63,
            min_data_in_leaf=50,
            feature_fraction=0.85,
            bagging_fraction=0.8,
            bagging_freq=1,
            random_state=args.seed,
            n_jobs=-1,
            label_gain=label_gain
        )

        ranker_model = None
        try:
            ranker = LGBMRanker(n_estimators=1200, **{k:v for k,v in ranker_params.items() if k not in ["metric","eval_at","label_gain"]})
            ranker.set_params(metric=ranker_params["metric"], eval_at=ranker_params["eval_at"], label_gain=ranker_params["label_gain"])
            ranker.fit(
                Xtr_rank, ytr_rank, group=tr_groups,
                eval_set=[(Xte_rank, yte_rank)],
                eval_group=[te_groups],
                callbacks=[
                    lgb.log_evaluation(period=50),
                    lgb.early_stopping(stopping_rounds=100)
                ]
            )
            ranker_model = ranker
            joblib.dump(ranker_model, outdir / "models" / "LGBMRanker.pkl")
        except TypeError as e:
            logging.warning(f"LGBMRanker.fit несовместим: {e}. Переходим на lgb.train.")
            lgb_train = lgb.Dataset(Xtr_rank, label=ytr_rank, group=tr_groups)
            lgb_valid = lgb.Dataset(Xte_rank, label=yte_rank, group=te_groups, reference=lgb_train)
            gbm = lgb.train(
                params=ranker_params,
                train_set=lgb_train,
                num_boost_round=2000,
                valid_sets=[lgb_valid],
                callbacks=[
                    lgb.log_evaluation(period=50),
                    lgb.early_stopping(stopping_rounds=100)
                ]
            )
            ranker_model = gbm
            gbm.save_model(str(outdir / "models" / "LGBMRanker.txt"))

        # Предсказания ранкера и калибровка
        if isinstance(ranker_model, LGBMRanker):
            pred_rank_test = ranker_model.predict(Xte_rank)
            train_scores = ranker_model.predict(Xtr_rank)
        else:
            pred_rank_test = ranker_model.predict(Xte_rank, num_iteration=getattr(ranker_model, "best_iteration", None))
            train_scores = ranker_model.predict(Xtr_rank, num_iteration=getattr(ranker_model, "best_iteration", None))

        unsort = np.argsort(test_df.index.get_indexer(te_idx_ordered))
        pred_rank = pred_rank_test[unsort]
        mm = MinMaxScaler()
        mm.fit(np.array(train_scores).reshape(-1, 1))
        pred_rank_cal = mm.transform(np.array(pred_rank).reshape(-1, 1)).ravel().clip(0, 1)
        joblib.dump(mm, outdir / "models" / "ranker_minmax.pkl")

        rows.append([
            "LGBMRanker (LambdaRank, per-day groups)",
            r2_score(y_te, pred_rank_cal),
            mean_absolute_error(y_te, pred_rank_cal),
            spearman_global(y_te, pred_rank_cal),
            spearman_per_day(test_df, y_te, pred_rank_cal)
        ])

        # ---------- 8) Отчёт ----------
        res_df = pd.DataFrame(rows, columns=["Model", "R2", "MAE", "Spearman_global", "Spearman_per_day"]) \
                 .sort_values("Spearman_per_day", ascending=False)
        print(res_df.to_string(index=False))
        res_df.to_csv(outdir / "results.csv", index=False, encoding="utf-8")
        log.info(f"Результаты сохранены: {outdir / 'results.csv'}")

        day_agg = (
            df_model.groupby("ph_day")
              .agg(day_N=("id", "size"),
                   day_min=("votesCount", "min"),
                   day_max=("votesCount", "max"))
              .reset_index()
              .sort_values("ph_day", ascending=False)
        )
        print("\nDaily stats (last 10 days):")
        print(day_agg.head(10).to_string(index=False))
        day_agg.to_csv(outdir / "daily_stats.csv", index=False, encoding="utf-8")

        # ---------- 9) Примеры предсказаний ----------
        sample_texts = [
            "AI app that generates pixel art avatars",
            "A devtool to auto-generate unit tests for Python projects",
        ]

        def make_new_features(texts):
            emb = sbert.encode(texts, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
            nums = np.zeros((len(texts), len(num_cols)), dtype=float)
            nums = scaler.transform(nums)
            return np.hstack([emb, nums])

        X_new = make_new_features(sample_texts)
        preds_out = {"samples": sample_texts, "preds": {}}

        # Регрессоры
        for name in reg_models.keys():
            safe_name = name.replace(" ", "_").replace("(", "_").replace(")", "_")
            mdl = joblib.load(outdir / "models" / f"{safe_name}.pkl")
            preds = np.clip(mdl.predict(X_new), 0.0, 1.0).tolist()
            print(f"{name}: {[round(float(p),3) for p in preds]}")
            preds_out["preds"][name] = preds

        # Ранкер
        if isinstance(ranker_model, LGBMRanker):
            rank_scores_new = ranker_model.predict(X_new)
        else:
            rank_scores_new = ranker_model.predict(X_new, num_iteration=getattr(ranker_model, "best_iteration", None))
        preds_rank_new = mm.transform(np.array(rank_scores_new).reshape(-1,1)).ravel().clip(0,1).tolist()
        print(f"LGBMRanker (LambdaRank): {[round(float(p),3) for p in preds_rank_new]}")
        preds_out["preds"]["LGBMRanker (LambdaRank)"] = preds_rank_new

        save_json(preds_out, outdir / "sample_predictions.json")
        log.info(f"Артефакты и модели сохранены в: {outdir.resolve()}")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("Критическая ошибка выполнения. Частично обученные артефакты уже могли быть сохранены выше.")
        raise
