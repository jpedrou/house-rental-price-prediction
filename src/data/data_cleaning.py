# ============================================================
# Data Cleaning
# ============================================================

# 1. Check NaN values
# 2. Check inconsistent data
# 3. Check Outliers

# ============================================================
# Libraries Import
# ============================================================

import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.impute import KNNImputer

# Setting plot configs
plt.rcParams["figure.figsize"] = (12, 5)
plt.rcParams["figure.dpi"] = 100
plt.style.use("ggplot")
sns.set_style("whitegrid")


# ============================================================
# Load Original Dataset
# ============================================================

original_df = pd.read_csv("../../data/raw/original_dataset.csv")

# Checking null values
original_df.isnull().mean() * 100

# Getting more data info
original_df.info()

# ============================================================
# Categorical Features
# ============================================================

cat_features = [
    col for col in original_df.columns if original_df[col].dtypes == "object"
]

for feature in cat_features:
    print(feature)
    print(original_df[feature].unique())
    print(original_df[feature].value_counts(normalize=True) * 100)
    print("\n")

# Making original_df copy
changed_df = original_df.copy()

# Cidade Column
changed_df["cidade"] = changed_df["cidade"].map(
    {
        "São Paulo": 0,
        "Rio de Janeiro": 1,
        "Belo Horizonte": 2,
        "Porto Alegre": 3,
        "Campinas": 4,
    }
)

# Estado Column
changed_df["estado"] = changed_df["estado"].map({"SP": 0, "RJ": 1, "MG": 2, "RS": 3})

# num_andares Column
sns.countplot(x=changed_df["num_andares"])
plt.xlabel("Número de andares")
plt.ylabel("")
plt.show()

count_distribution = changed_df.loc[
    changed_df["num_andares"] != "-", "num_andares"
].value_counts(normalize=True)

value = (
    changed_df.loc[changed_df["num_andares"] == "-", "num_andares"]
    .value_counts()
    .values[0]
)

changed_df.loc[changed_df["num_andares"] == "-", "num_andares"] = np.nan

for key, number in count_distribution.items():
    dis = math.ceil(value * number)
    changed_df["num_andares"].fillna(key, limit=dis, inplace=True)
    value -= dis

changed_df["num_andares"].fillna(changed_df["num_andares"].mode()[0], inplace=True)


changed_df["num_andares"].isnull().sum()
changed_df["num_andares"].dtypes
changed_df["num_andares"] = changed_df["num_andares"].astype(int)

plt.figure()
sns.countplot(x=changed_df["num_andares"])
plt.xlabel("Número de andares")
plt.ylabel("")
plt.savefig("../../reports/numero_andares_distribution.jpg")
plt.show()


# aceita_animais Column
changed_df["aceita_animais"].unique()
plt.figure()
sns.countplot(x=changed_df["aceita_animais"], width=0.5)
plt.xlabel("Distribution aceita_animais")
plt.ylabel("")
plt.savefig("../../reports/aceita_animais_distribution.jpg")
plt.show()
changed_df["aceita_animais"] = changed_df["aceita_animais"].map(
    {"acept": 1, "not acept": 0}
)


# mobilia Column
changed_df["mobilia"].unique()

plt.figure()
sns.countplot(x=changed_df["mobilia"], width=0.5)
plt.xlabel("Distribution mobilia")
plt.ylabel("")
plt.savefig("../../reports/mobilia_distribution.jpg")
plt.show()

changed_df["mobilia"] = changed_df["mobilia"].map({"furnished": 1, "not furnished": 0})


# ============================================================
# Numeric Features
# ============================================================

changed_df2 = changed_df.copy()

num_features = [
    col
    for col in original_df.columns
    if original_df[col].dtypes == int or original_df[col].dtypes == float
]


for feature in num_features:

    fig, ax = plt.subplots(nrows=1, ncols=2)
    sns.histplot(ax=ax[0], x=changed_df2[feature], kde=True)
    sns.boxplot(ax=ax[1], y=changed_df2[feature])
    ax[0].set_ylabel("")
    ax[1].set_ylabel("")
    ax[1].set_xlabel(feature)
    plt.show()


# ============================================================
# Handling Outliers
# ============================================================

df_with_no_outliers = changed_df2.copy()

df_with_no_outliers[num_features].describe()

df_with_no_outliers["area"] = np.where(
    df_with_no_outliers["area"] > 2000, np.nan, df_with_no_outliers["area"]
)

df_with_no_outliers["valor_aluguel"] = np.where(
    df_with_no_outliers["area"].isna(), np.nan, df_with_no_outliers["valor_aluguel"]
)

df_with_no_outliers["valor_condominio"] = np.where(
    df_with_no_outliers["area"].isna(), np.nan, df_with_no_outliers["valor_condominio"]
)

df_with_no_outliers["valor_iptu"] = np.where(
    df_with_no_outliers["area"].isna(), np.nan, df_with_no_outliers["valor_iptu"]
)

df_with_no_outliers["valor_aluguel"] = np.where(
    df_with_no_outliers["valor_aluguel"] > 40000,
    np.nan,
    df_with_no_outliers["valor_aluguel"],
)

df_with_no_outliers["valor_iptu"] = np.where(
    df_with_no_outliers["valor_iptu"] > 30000, np.nan, df_with_no_outliers["valor_iptu"]
)

# Checking if the outliers were changed to NaN
nan_index = df_with_no_outliers[df_with_no_outliers.isnull().any(axis=1)].index

# KNN Imputation
imputer = KNNImputer(n_neighbors=10, weights="distance")

df_with_no_outliers[["area", "valor_aluguel", "valor_condominio", "valor_iptu"]] = (
    imputer.fit_transform(
        df_with_no_outliers[["area", "valor_aluguel", "valor_condominio", "valor_iptu"]]
    )
)

df_with_no_outliers.loc[nan_index]

for feature in num_features:

    fig, ax = plt.subplots(nrows=2, ncols=2, gridspec_kw={"hspace": 0.4})
    sns.histplot(ax=ax[0][0], x=original_df[feature], kde=True)
    sns.boxplot(ax=ax[0][1], y=original_df[feature])
    sns.histplot(ax=ax[1][0], x=df_with_no_outliers[feature], kde=True, color="blue")
    sns.boxplot(ax=ax[1][1], y=df_with_no_outliers[feature], color="blue")
    ax[0][0].set_ylabel("")
    ax[0][0].set_xlabel(f"Original {feature}")
    ax[1][0].set_ylabel("")
    ax[1][0].set_xlabel(f"{feature} with no outliers")
    ax[0][1].set_xlabel(f"Original {feature}")
    ax[1][1].set_xlabel(f"{feature} with no outliers")
    plt.savefig(f"../../reports/{feature}_distribution_comparison.jpg")
    plt.show()

# ============================================================
# Export new dataset
# ============================================================

df_with_no_outliers.to_csv('../../data/processed/df_with_no_outliers.csv', index = None)
