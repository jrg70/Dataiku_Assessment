# preprocess.py
import pandas as pd

def preprocess_data(df, target_col="income"):
    """Clean the dataset:
       1. Remove duplicate rows (ignoring target_col)
       2. Drop columns with >40% NaNs, previously identified as migration code-change in msa, migration code-change in reg, migration code-move within reg, migration prev res in sunbelt 
       3. Drop rows with any remaining NaNs, about 4% of data which is acceptable for this use case as data loss is minimal.
    """
    # 1.
    df_no_dup = df.drop_duplicates(subset=df.columns.difference([target_col]))
    
    # 2.
    threshold = 0.4
    na_ratio = df_no_dup.isna().mean()
    df_reduced = df_no_dup.drop(columns=na_ratio[na_ratio > threshold].index)
    
    # 3.
    df_clean = df_reduced.dropna()
    
    return df_clean
