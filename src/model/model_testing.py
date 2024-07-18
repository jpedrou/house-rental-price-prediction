# ===============================================================
# Testing Model
# ===============================================================


# ===============================================================
# Libraries Import
# ===============================================================

import math
import pandas as pd
import joblib as jb
import numpy as np
import matplotlib.pyplot as plt

from sklearn.impute import KNNImputer
from sklearn.metrics import r2_score, mean_squared_error


plt.rcParams["figure.figsize"] = (15, 10)
plt.rcParams["figure.dpi"] = 100
plt.style.use("fivethirtyeight")

# ===============================================================
# Load Test Dataset
# ===============================================================

test_df = pd.read_csv("../../data/raw/test_dataset.csv")

# ===============================================================
# Functions to transform the data
# ===============================================================


def preprocess_data(df):
    tmp = df.copy()

    # Cidade Column
    tmp["cidade"] = tmp["cidade"].map(
        {
            "São Paulo": 0,
            "Rio de Janeiro": 1,
            "Belo Horizonte": 2,
            "Porto Alegre": 3,
            "Campinas": 4,
        }
    )

    # Estado Column
    tmp["estado"] = tmp["estado"].map({"SP": 0, "RJ": 1, "MG": 2, "RS": 3})

    # num_andares Column
    count_distribution = tmp.loc[tmp["num_andares"] != "-", "num_andares"].value_counts(
        normalize=True
    )

    value = tmp.loc[tmp["num_andares"] == "-", "num_andares"].value_counts().values[0]

    tmp.loc[tmp["num_andares"] == "-", "num_andares"] = np.nan

    for key, number in count_distribution.items():
        dis = math.ceil(value * number)
        tmp["num_andares"].fillna(key, limit=dis, inplace=True)
        value -= dis

    tmp["num_andares"].fillna(tmp["num_andares"].mode()[0], inplace=True)

    # aceita_animais Column
    tmp["aceita_animais"] = tmp["aceita_animais"].map({"acept": 1, "not acept": 0})

    # mobilia Column
    tmp["mobilia"] = tmp["mobilia"].map({"furnished": 1, "not furnished": 0})

    return tmp


def handle_outliers(df):
    tmp = df.copy()

    tmp["area"] = np.where(tmp["area"] > 2000, np.nan, tmp["area"])

    tmp["valor_aluguel"] = np.where(tmp["area"].isna(), np.nan, tmp["valor_aluguel"])

    tmp["valor_condominio"] = np.where(
        tmp["area"].isna(), np.nan, tmp["valor_condominio"]
    )

    tmp["valor_iptu"] = np.where(tmp["area"].isna(), np.nan, tmp["valor_iptu"])

    tmp["valor_aluguel"] = np.where(
        tmp["valor_aluguel"] > 40000,
        np.nan,
        tmp["valor_aluguel"],
    )

    tmp["valor_iptu"] = np.where(tmp["valor_iptu"] > 30000, np.nan, tmp["valor_iptu"])

    tmp["num_andares"] = tmp["num_andares"].astype(int)

    tmp["num_andares"] = np.where(tmp["num_andares"] > 32, np.nan, tmp["num_andares"])

    imputer = KNNImputer(n_neighbors=10, weights="distance")

    tmp[["area", "valor_aluguel", "valor_condominio", "valor_iptu", "num_andares"]] = (
        imputer.fit_transform(
            tmp[
                [
                    "area",
                    "valor_aluguel",
                    "valor_condominio",
                    "valor_iptu",
                    "num_andares",
                ]
            ]
        )
    )

    return tmp


# ===============================================================
# Transforming the test_set
# ===============================================================

test_set = preprocess_data(test_df)
test_set = handle_outliers(test_set)

X_test = test_set.drop(["valor_aluguel"], axis=1)
y_test = test_set["valor_aluguel"]

# ===============================================================
# Making Predictions
# ===============================================================

# Loading Model
model = jb.load("../model/model.pkl")

y_test_pred = model.predict(X_test)

print("R2_score:", r2_score(y_test, y_test_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_test_pred)))


# ===============================================================
# Plotting Predictions
# ===============================================================

plt.hist(y_test, label="Real")
plt.hist(y_test_pred, label="Predictions")
plt.legend()
plt.show()
