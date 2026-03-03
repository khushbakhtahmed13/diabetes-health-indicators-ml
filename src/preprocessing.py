from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np


def build_preprocessor(numeric_cols, ord_cols, ohe_cols):
    numeric_pipeline = Pipeline([
        ('log', FunctionTransformer(np.log1p, feature_names_out='one-to-one')),
        ('scaler', RobustScaler())

    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, numeric_cols),
            ('ord', OrdinalEncoder(categories=[["No formal", "Highschool", "Graduate", "Postgraduate"],
                                               ["Low", "Lower-Middle", "Middle", "Upper-Middle", "High"]],
                                   handle_unknown='use_encoded_value', unknown_value=-1), ord_cols),
            ('ohe', OneHotEncoder(drop="first", handle_unknown="ignore"), ohe_cols)
        ])

    return preprocessor