# =============================================================== #
#                                                                 #                                              
#                                                                 #
#                 Machine Learning Model for predict              #
#                 predict the rent value of a house               #
#                                                                 #
#                                                                 #
# =============================================================== #

# 1. Select the best features group
# 2. Select a Machine Learning model
# 3. Optimize the hyperparameters

# ===============================================================
# Libraries Import
# ===============================================================

import optuna
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

from optuna.samplers import TPESampler
from catboost import CatBoostRegressor


# ===============================================================
# Load Data
# ===============================================================

df = pd.read_csv("../../data/processed/df_with_new_features.csv")

# Split into X and y

X = df.drop(["valor_aluguel"], axis=1)
y = df["valor_aluguel"]

# Selecting different variables sets
original_features = list(X.iloc[:, :11].columns)
pca_features = [
    feature for feature in X.columns if feature.endswith(("pca_0", "pca_1"))
]
cluster_feature = ["cluster"]

# Combining features

set1 = list(set(original_features))
set2 = list(set(original_features + pca_features))
set3 = list(set(pca_features + cluster_feature))
set4 = list(set(original_features + pca_features + cluster_feature))

sets = {"set1": set1, "set2": set2, "set3": set3, "set4": set4}


# ===============================================================
# Base Model
# ===============================================================


def return_train_results(model, vars):
    results = list()
    for key, value in vars.items():
        X_train, X_dev, y_train, y_dev = train_test_split(
            X[value], y, test_size=0.1, random_state=0
        )

        model.fit(X_train, y_train)
        # Train Prediction
        y_train_pred = model.predict(X_train)

        # Train Metrics
        rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
        r2_score_train = r2_score(y_train, y_train_pred)

        # Dev Prediction
        y_dev_pred = model.predict(X_dev)

        # Dev Metrics
        rmse_dev = np.sqrt(mean_squared_error(y_dev, y_dev_pred))
        r2_score_dev = r2_score(y_dev, y_dev_pred)

        metrics = {
            key: {
                "train_results": {"train_rmse": rmse_train, "train_r2": r2_score_train},
                "dev_results": {"dev_rmse": rmse_dev, "dev_r2": r2_score_dev},
            }
        }

        results.append(metrics)

    flat_results = list()
    for r in results:
        for set, result in r.items():
            flat_results.append(
                {
                    "set": set,
                    "train_rmse": result["train_results"]["train_rmse"],
                    "train_r2": result["train_results"]["train_r2"],
                    "dev_rmse": result["dev_results"]["dev_rmse"],
                    "dev_r2": result["dev_results"]["dev_r2"],
                }
            )
    results_df = pd.DataFrame(flat_results)

    return results_df


# Nearest Neighbours Regressor
knn = KNeighborsRegressor(n_neighbors=20, n_jobs=-1)
return_train_results(knn, sets)

# ExtraTress Regressor
ex = ExtraTreesRegressor(n_estimators=100, n_jobs=-1, random_state=0)
return_train_results(ex, sets)

# Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=0)
return_train_results(rf, sets)

# Catboost Regressor
ct = CatBoostRegressor(verbose=0, random_state=0)
return_train_results(ct, sets)


# ===============================================================
# Hyperparameters Optimization
# ===============================================================


def objective(trial, X_train, X_dev, y_train, y_dev):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1.0, log=True),
        "depth": trial.suggest_int("depth", 4, 12),
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 10.0),
        "grow_policy": trial.suggest_categorical(
            "grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]
        ),
    }

    model = CatBoostRegressor(**params, verbose=False)
    model.fit(X_train, y_train)

    y_dev_pred = model.predict(X_dev)

    rmse = np.sqrt(mean_squared_error(y_dev, y_dev_pred))

    return rmse


results = list()
for name, value in sets.items():
    X_train, X_dev, y_train, y_dev = train_test_split(
        X[value], y, test_size=0.1, random_state=0
    )

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=1001),
        study_name=f"Catboost Regressor Optimizer on {name}",
    )

    def objective_with_data(trial):
        return objective(trial, X_train, X_dev, y_train, y_dev)

    study.optimize(objective_with_data, n_trials=200)
    results.append(
        {"set": name, "best_rmse_value": study.best_value, "best_params": study.best_params}
    )

results_df = pd.DataFrame(results)