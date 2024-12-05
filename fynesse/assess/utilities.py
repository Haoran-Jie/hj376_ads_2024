import numpy as np
import pandas as pd
import yaml
import warnings


def calculate_corr(y, y_pred):
    return np.corrcoef(y, y_pred)[0, 1]

def load_credentials(yaml_file = "../credentials.yaml"):
    with open(yaml_file) as file:
        credentials = yaml.safe_load(file)
    return credentials['username'], credentials['password'], credentials['url'], credentials['port']

def read_sql_ignoring_warnings(query, con, *args, **kwargs):
    """Wrapper for pandas.read_sql that suppresses UserWarnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.read_sql(query, con, *args, **kwargs)