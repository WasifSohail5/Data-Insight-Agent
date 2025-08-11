import pandas as pd
import io
import openpyxl
def read_file(file) -> pd.DataFrame:
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith('.xlsx'):
        return pd.read_excel(file)
    elif file.name.endswith('.xls'):
        return pd.read_excel(file, engine='xlrd')
    elif file.name.endswith('.json'):
        return pd.read_json(file)
    elif file.name.endswith('.parquet'):
        return pd.read_parquet(file)
    elif file.name.endswith('.feather'):
        return pd.read_feather(file)
    elif file.name.endswith('.pickle') or file.name.endswith('.pkl'):
        return pd.read_pickle(file)
    elif file.name.endswith('.tsv') or file.name.endswith('.txt'):
        return pd.read_csv(file, sep='\t')
    elif file.name.endswith('.html'):
        return pd.read_html(file)[0]
    else:
        raise ValueError("Unsupported file format: {}".format(file.name))

def to_csv(df: pd.DataFrame)-> bytes:
    return df.to_csv(index=False).encode('utf-8')
