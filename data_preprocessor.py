import numpy as np
import pandas as pd

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df =df.drop_duplicates()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.strip().str.lower()
        elif np.issubdtype(df[col].dtype, np.number):
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col]=df[col].fillna(df[col].median())

    return df
def get_basic_stats(df: pd.DataFrame) -> pd.DataFrame:
    stats = {
        'column': [],
        'mean': [],
        'median': [],
        'mode': [],
        'std_dev': []
    }

    for col in df.select_dtypes(include=[np.number]).columns:
        stats['column'].append(col)
        stats['mean'].append(df[col].mean())
        stats['median'].append(df[col].median())
        stats['mode'].append(df[col].mode()[0])
        stats['std_dev'].append(df[col].std())
    info={
        "Shape": df.shape,
        "Columns": df.columns.tolist(),
        "Data Types": df.dtypes.to_dict(),
        "Missing Values": df.isnull().sum().to_dict(),
    }
    return pd.DataFrame(stats)