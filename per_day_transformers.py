# per_day_analysis.py — группировка по дням, ранги в дне и обучение моделей (SBERT + числа + теги)
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import MaxAbsScaler

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# ---------- 1) Загрузка данных ----------
df = pd.read_excel("output.xlsx").copy()

# Базовая чистка/типы
df["votesCount"] = pd.to_numeric(df.get("votesCount"), errors="coerce").fillna(0).astype(int)
df["commentsCount"] = pd.to_numeric(df.get("commentsCount"), errors="coerce").fillna(0).astype(int)
df["createdAt"] = pd.to_datetime(df.get("createdAt"), errors="coerce")

# День публикации (если нет даты — помечаем 'unknown', такие строки потом можно дропнуть)
df["ph_day"] = df["createdAt"].dt.date.astype("string")
df.loc[df["createdAt"].isna(), "ph_day"] = "unknown"

# ---------- 2) Дневные агрегации и ранги ----------
# Сколько постов в дне
df["day_N"] = df.groupby("ph_day")["id"].transform("size").astype(int)

# Мин/макс голосов в дне
df["day_votes_min"] = df.groupby("ph_day")["votesCount"].transform("min").astype(int)
df["day_votes_max"] = df.groupby("ph_day")["votesCount"].transform("max").astype(int)

# Место в своём дне: 1.0 — лучший (больше голосов), тай-брейки — средний ранг
df["day_rank"] = df.groupby("ph_day")["votesCount"].rank(method="average", ascending=False)

# Перцентиль силы внутри дня в [0,1], где ~1 — топ дня
df["day_percentile"] = (df["day_N"] - df["day_rank"] + 0.5) / df["day_N"].clip(lower=1)

# Для справки: min–max нормировка в пределах дня (не используем как таргет по умолчанию)
rng = (df["day_votes_max"] - df["day_votes_min"]).replace(0, 1)
df["day_minmax"] = (df["votesCount"] - df["day_votes_min"]) / rng

# Если есть строки с ph_day = 'unknown' (нет даты) — можно удалить из обучения
mask_known_day = df["ph_day"] != "unknown"
df_model = df.loc[mask_known_day].copy()

# ---------- 3) Текст и цель ----------
# базовые числовые фичи
df_model["tagline_len"] = df_model["tagline"].fillna("").str.len()
df_model["desc_len"]    = df_model["description"].fillna("").str.len()

# гарантируем наличие числовых колонок (если их нет в файле — создаём с нулями)
for col in ["img_count", "video_count", "topics_count"]:
    if col not in df_model.columns:
        df_model[col] = 0

# день недели (циклическое кодирование sin/cos)
df_model["dow"] = df_model["createdAt"].dt.dayofweek
df_model["dow_sin"] = np.sin(2 * np.pi * df_model["dow"] / 7)
df_model["dow_cos"] = np.cos(2 * np.pi * df_model["dow"] / 7)

# текст = tagline + description + topics_str
if "topics_str" not in df_model.columns:
    df_model["topics_str"] = ""
df_model["topics_str"] = df_model["topics_str"].fillna("")

def join_text(row):
    return f"{row.get('tagline','')} {row.get('description','')} {row.get('topics_str','')}".strip()

df_model["text_all"] = df_model.apply(join_text, axis=1)

y_all = df_model["day_percentile"].astype(float)

# Сплит (оставляем как у тебя — random split; при желании заменишь на time-based)
X_tr_txt, X_te_txt, y_tr, y_te = train_test_split(
    df_model["text_all"], y_all, test_size=0.2, random_state=42
)

# Числовые признаки + масштабирование
num_cols = ["img_count", "video_count", "topics_count", "tagline_len", "desc_len",
            "dow_sin", "dow_cos", "day_N"]
X_tr_num = df_model.loc[X_tr_txt.index, num_cols].to_numpy(dtype=float)
X_te_num = df_model.loc[X_te_txt.index, num_cols].to_numpy(dtype=float)

scaler = MaxAbsScaler()
X_tr_num = scaler.fit_transform(X_tr_num)
X_te_num = scaler.transform(X_te_num)

# ---------- 3.1) Fine-tune SBERT как bi-encoder на парах внутри дня ----------
sbert = SentenceTransformer("all-MiniLM-L6-v2")

# Гиперпараметры формирования пар
TOP_K = 5         # сколько брать лучших из дня
BOT_K = 30          # сколько брать худших из дня
MIN_DELTA = 0.1    # минимальная разница перцентилей, чтобы пара считалась «сильной»

train_examples = []
# Формируем пары ТОЛЬКО из train-части (без утечки в test)
train_df = df_model.loc[X_tr_txt.index]

for _, g in train_df.groupby("ph_day", sort=False):
    if len(g) < 3:
        continue
    # берём top/bottom по day_percentile
    gt = g.nlargest(TOP_K, "day_percentile")
    gb = g.nsmallest(BOT_K, "day_percentile")

    # 1) Негативные пары: top vs bottom → ДОЛЖНЫ БЫТЬ НЕПОХОЖИ (label = 0.0)
    for _, a in gt.iterrows():
        for _, b in gb.iterrows():
            if (a["day_percentile"] - b["day_percentile"]) < MIN_DELTA:
                continue
            train_examples.append(
                InputExample(texts=[a["text_all"], b["text_all"]], label=0.0)
            )

    # 2) Позитивные пары: внутри топа → ДОЛЖНЫ БЫТЬ ПОХОЖИ (label = 1.0)
    gt_list = gt["text_all"].tolist()
    for i in range(len(gt_list) - 1):
        train_examples.append(
            InputExample(texts=[gt_list[i], gt_list[i+1]], label=1.0)
        )

    # 3) Позитивные пары: внутри низа → ДОЛЖНЫ БЫТЬ ПОХОЖИ (label = 1.0)
    gb_list = gb["text_all"].tolist()
    for i in range(len(gb_list) - 1):
        train_examples.append(
            InputExample(texts=[gb_list[i], gb_list[i+1]], label=1.0)
        )

if len(train_examples) > 0:
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)
    train_loss = losses.CosineSimilarityLoss(model=sbert)
    # Можно слегка увеличить эпохи и задать явный LR для стабильности
    sbert.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=2,
        warmup_steps=min(1000, max(10, len(train_dataloader)//10)),
        show_progress_bar=True
        # optimizer_params={"lr": 2e-5},  # <- по желанию раскомментируй
    )
else:
    print("Warning: не удалось сформировать пары для обучения SBERT — пропускаю fine-tune.")

# ---------- 3.2) Кодируем тексты эмбеддингами и склеиваем с числами ----------
# Нормализуем эмбеддинги — это помогает линейным моделям/GBDT
X_tr_emb = sbert.encode(X_tr_txt.tolist(), batch_size=256, normalize_embeddings=True, show_progress_bar=False)
X_te_emb = sbert.encode(X_te_txt.tolist(), batch_size=256, normalize_embeddings=True, show_progress_bar=False)

# Склейка: эмбеддинги (dense) + числовые (dense)
X_tr = np.hstack([X_tr_emb, X_tr_num])
X_te = np.hstack([X_te_emb, X_te_num])

# для совместимости с твоим циклом обучения
X_tr_dense = X_tr
X_te_dense = X_te

# ---------- 4) Набор моделей ----------
models = {
    "Ridge (alpha=3)": Ridge(alpha=3.0, fit_intercept=True),
    "SGD (MSE, l2)": SGDRegressor(
        loss="squared_error", penalty="l2", alpha=1e-4, max_iter=2000, random_state=42
    ),
    "SGD (Huber)": SGDRegressor(
        loss="huber", penalty="l2", alpha=1e-4, max_iter=2000, random_state=42
    ),
    "SGD (ElasticNet)": SGDRegressor(
        loss="squared_error", penalty="elasticnet", l1_ratio=0.15,
        alpha=1e-4, max_iter=2000, random_state=42
    ),
    "RandomForest(dense)": RandomForestRegressor(
        n_estimators=400, max_depth=None, n_jobs=-1, random_state=42
    ),
    "GradBoost(dense)": GradientBoostingRegressor(random_state=42),
    "XGBoost": XGBRegressor(
        n_estimators=800, max_depth=8, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, tree_method="hist", random_state=42, n_jobs=-1
    ),
    "LightGBM": LGBMRegressor(
        objective="regression",
        n_estimators=1500, learning_rate=0.04,
        num_leaves=63, min_data_in_leaf=100,
        feature_fraction=0.85,
        bagging_fraction=0.8, bagging_freq=1,
        reg_lambda=2.0,
        force_col_wise=True,
        random_state=42, n_jobs=-1
    ),
}

# ---------- 5) Обучение и метрики ----------
rows = []
for name, model in models.items():
    # теперь все данные — dense; но оставляем твой интерфейс «dense/не dense» как есть
    if "dense" in name:
        model.fit(X_tr_dense, y_tr)
        y_pred = model.predict(X_te_dense)
    else:
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

    r2 = r2_score(y_te, y_pred)
    mae = mean_absolute_error(y_te, y_pred)
    spear = float(spearmanr(y_te, y_pred).correlation)
    rows.append([name, r2, mae, spear])

res_df = pd.DataFrame(rows, columns=["Model", "R2", "MAE", "Spearman"]) \
         .sort_values("Spearman", ascending=False)
print(res_df.to_string(index=False))

# ---------- 6) Короткая сводка по дням ----------
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

# ---------- 7) Предикты на новом тексте ----------
sample_texts = [
    "AI app that generates pixel art avatars",
    "A devtool to auto-generate unit tests for Python projects",
]

# Эмбеддинги новых текстов + нули для числовых, но через тот же scaler
X_new_emb = sbert.encode(sample_texts, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
X_new_num = np.zeros((len(sample_texts), len(num_cols)), dtype=float)
X_new_num = scaler.transform(X_new_num)
X_new = np.hstack([X_new_emb, X_new_num])

print("\nPredictions (estimated day-percentile in [0,1]) on new texts:")
for name, model in models.items():
    preds = model.predict(X_new)
    preds = np.clip(preds, 0.0, 1.0)
    print(f"{name}: {[round(float(p),3) for p in preds]}")
