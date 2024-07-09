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


# aceita_animais Column
changed_df["aceita_animais"].unique()
changed_df["aceita_animais"] = changed_df["aceita_animais"].map(
    {"acept": 1, "not acept": 0}
)

# mobilia Column
changed_df["mobilia"].unique()
changed_df["mobilia"] = changed_df["mobilia"].map({"furnished": 1, "not furnished": 0})


# ============================================================
# Numeric Features
# ============================================================
num_features = [
    col
    for col in original_df.columns
    if original_df[col].dtypes == int or original_df[col].dtypes == float
]
