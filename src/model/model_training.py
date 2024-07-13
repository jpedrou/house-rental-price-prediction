# ===============================================================
# Training Selected Model
# ===============================================================


# ===============================================================
# Libraries Import
# ===============================================================

import joblib as jb
import pandas as pd
import numpy as np

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

from catboost import CatBoostRegressor

# ===============================================================
# Load Data / Results Comparison
# ===============================================================

df = pd.read_csv("../../data/processed/df_with_new_features.csv")

X = df.drop(["valor_aluguel"], axis=1)
y = df["valor_aluguel"]

original_features = list(X.iloc[:, :11].columns)

# ===============================================================
# Split Data
# ===============================================================

X_train, X_dev, y_train, y_dev = train_test_split(
    X[original_features], y, test_size=0.2, random_state=0
)


# ===============================================================
# Model Configuration
# ===============================================================

results = pd.read_csv("../../reports/model_comparison.csv")
config = eval(results["best_params"][0])

model = CatBoostRegressor(**config, verbose=0, random_state=0)

# ===============================================================
# Model Training
# ==============================================================

model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)

# Train Metrics
train_r2_score = r2_score(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

y_dev_pred = model.predict(X_dev)

# Dev Metrics
dev_r2_score = r2_score(y_dev, y_dev_pred)
dev_rmse = np.sqrt(mean_squared_error(y_dev, y_dev_pred))


# ===============================================================
# Export Model
# ==============================================================

jb.dump(model, '../model/model.pkl')