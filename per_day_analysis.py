# per_day_analysis.py — группировка по дням, ранги в дне и обучение моделей (текст + числа + теги)
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import MaxAbsScaler   # <-- ДОБАВЛЕНО

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from scipy.sparse import hstack, csr_matrix

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

# день недели (sin-кодировка)
df_model["dow"] = df_model["createdAt"].dt.dayofweek
df_model["dow_sin"] = np.sin(2 * np.pi * df_model["dow"] / 7)
df_model["dow_cos"] = np.cos(2 * np.pi * df_model["dow"] / 7)

# текст = tagline + description + topics_str
if "topics_str" not in df_model.columns:
    df_model["topics_str"] = ""
df_model["topics_str"] = df_model["topics_str"].fillna("")

text_all = (
    df_model["tagline"].fillna("") + " " +
    df_model["description"].fillna("") + " " +
    df_model["topics_str"]
).str.strip()

y_all = df_model["day_percentile"].astype(float)

# Сплит
X_tr_txt, X_te_txt, y_tr, y_te = train_test_split(
    text_all, y_all, test_size=0.2, random_state=42
)

# TF-IDF
tfidf = TfidfVectorizer(
    lowercase=True, stop_words="english",
    ngram_range=(1,2),
    min_df=10,           # было 5
    max_df=0.7,          # было 0.8
    max_features=80000,  # можно дать больше верхнюю границу
    norm="l2", sublinear_tf=True,
)

X_tr_text = tfidf.fit_transform(X_tr_txt)
X_te_text = tfidf.transform(X_te_txt)

# приклеиваем числовые фичи + МАСШТАБИРУЕМ ИХ (ВАЖНО)
num_cols = ["img_count", "video_count", "topics_count", "tagline_len", "desc_len", "dow_sin", "dow_cos","day_N"]
X_tr_num = df_model.loc[X_tr_txt.index, num_cols].to_numpy(dtype=float)
X_te_num = df_model.loc[X_te_txt.index, num_cols].to_numpy(dtype=float)

scaler = MaxAbsScaler()                 # масштабируем только числовые колонки
X_tr_num = scaler.fit_transform(X_tr_num)
X_te_num = scaler.transform(X_te_num)

X_tr = hstack([X_tr_text, csr_matrix(X_tr_num)])
X_te = hstack([X_te_text, csr_matrix(X_te_num)])

# dense для леса/градбуста
X_tr_dense = X_tr.toarray()
X_te_dense = X_te.toarray()

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
        num_leaves=63,
        min_data_in_leaf=100,         # было 15
        feature_fraction=0.85,
        bagging_fraction=0.8, bagging_freq=1,
        reg_lambda=2.0,               # L2-рег
        force_col_wise=True,
        random_state=42, n_jobs=-1
    ),
}

# ---------- 5) Обучение и метрики ----------
rows = []
for name, model in models.items():
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
X_new_text = tfidf.transform(sample_texts)

# для новых «свободных» текстов числовых фич нет → нули, НО ТОЖЕ ПРОГОНЯЕМ ЧЕРЕЗ scaler
X_new_num = np.zeros((len(sample_texts), len(num_cols)), dtype=float)
X_new_num = scaler.transform(X_new_num)          # <-- ВАЖНО, чтобы масштаб совпадал с train
X_new = hstack([X_new_text, csr_matrix(X_new_num)])

print("\nPredictions (estimated day-percentile in [0,1]) on new texts:")
for name, model in models.items():
    if "dense" in name:
        preds = model.predict(X_new.toarray())
    else:
        preds = model.predict(X_new)
    preds = np.clip(preds, 0.0, 1.0)
    print(f"{name}: {[round(float(p),3) for p in preds]}")
