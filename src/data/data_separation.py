# ============================================================
# Data Separation
# ============================================================

import pandas as pd

# ============================================================
# Load Data
# ============================================================

df = pd.read_csv('../../data/raw/original_dataset.csv')

# ============================================================
# Split into Train and Test base
# ============================================================

test_df = df.sample(n=2000, random_state=0)
df = df.drop(test_df.index, axis = 0)

df.reset_index(drop = True, inplace=True)
test_df.reset_index(drop = True, inplace=True)

# ============================================================
# Export Datasets
# ============================================================

df.to_csv('../../data/raw/original_dataset.csv', index = None)
test_df.to_csv('../../data/raw/test_dataset.csv', index = None)