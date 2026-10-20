"""All learned transformations live inside training pipelines."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.data.loader import CATEGORICAL


def make_preprocessor(columns):
    cats = [c for c in columns if c in CATEGORICAL]
    nums = [c for c in columns if c not in cats]
    return ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("scale", StandardScaler())]), nums),
        ("category", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cats)
    ], verbose_feature_names_out=True)
