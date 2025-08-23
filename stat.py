import pandas as pd
import numpy as np
df = pd.read_excel("output.xlsx")
df["desc_len^2"] = df["desc_len"] ** 2
df["tagline_len^2"] = df["tagline_len"] ** 2
df["topics_count^2"] = df["topics_count"] ** 2 
df["img_count^2"] = df["img_count"] ** 2
df["video_count^2"] = df["video_count"] ** 2
df["tagline_len*desc_len"] = df["tagline_len"] * df["desc_len"]
df["tagline_len*topics_count"] = df["tagline_len"] * df["topics_count"]
X = df[["tagline_len", "desc_len", "topics_count", "img_count", "video_count", "tagline_len^2", "desc_len^2", "topics_count^2", "img_count^2"]].to_numpy()
y = df["votesCount"].to_numpy()

X = (X - X.mean(axis=0)) / X.std(axis=0, ddof=0)

ones = np.ones((X.shape[0], 1)) 
X_aug = np.hstack([ones, X])

w = np.linalg.pinv(X_aug.T @ X_aug) @ X_aug.T @ y

print(w)

y_hat = X_aug @ w
r = y - y_hat

r_mean = r.mean()
mse = np.mean(r**2)
rmse = np.sqrt(mse)
sse = np.sum(r**2)
sst = np.sum((y - y.mean())**2)
r2 = 1 - sse/sst

print(r2)