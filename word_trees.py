import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from scipy.stats import spearmanr

# 1) Данные
df = pd.read_excel("output.xlsx")

# 2) Тексты и целевая переменная
text_all = (df["tagline"].fillna("") + " " + df["description"].fillna("")).str.strip()
y_all = np.log1p(df["votesCount"].astype(float))  # log(1 + votes)

# 3) train/test ДО векторизации
X_tr_txt, X_te_txt, y_tr, y_te = train_test_split(
    text_all, y_all, test_size=0.2, random_state=42
)

# 4) TF-IDF
tfidf = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.8,
    max_features=50000,
    norm="l2",
    sublinear_tf=True,
)
X_tr = tfidf.fit_transform(X_tr_txt)   # sparse CSR
X_te = tfidf.transform(X_te_txt)       # sparse CSR

# dense для тех, кому нужно
X_tr_dense = X_tr.toarray()
X_te_dense = X_te.toarray()

# 5) Модели
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
        n_estimators=1000, max_depth=-1, learning_rate=0.05, num_leaves=20,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
    ),
}

# 6) Обучение и оценка
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
    spear = spearmanr(y_te, y_pred).correlation
    rows.append([name, r2, mae, spear])

res_df = pd.DataFrame(rows, columns=["Model", "R2", "MAE", "Spearman"]) \
         .sort_values("Spearman", ascending=False)
print(res_df.to_string(index=False))

# 7) Инференс на новом тексте
new_text = ["AI app that generates pixel art avatars"]
X_new = tfidf.transform(new_text)

print("\nPredictions on new text:")
for name, model in models.items():
    if "dense" in name:
        pred = model.predict(X_new.toarray())[0]
    else:
        pred = model.predict(X_new)[0]
    print(f"{name}: {pred:.4f}")
