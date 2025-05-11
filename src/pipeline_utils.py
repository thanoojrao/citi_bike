import lightgbm as lgb
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler


# Function to calculate the average rides over the last 4 weeks
def average_rides_last_4_weeks(X: pd.DataFrame) -> pd.DataFrame:
    last_4_weeks_columns = [
        f"rides_t-{7*4}",  # 1 week ago
        f"rides_t-{14*4}",  # 2 weeks ago
        f"rides_t-{21*4}",  # 3 weeks ago
        f"rides_t-{28*4}",  # 4 weeks ago
    ]

    # Ensure the required columns exist in the DataFrame
    for col in last_4_weeks_columns:
        if col not in X.columns:
            raise ValueError(f"Missing required column: {col}")

    # Calculate the average of the last 4 weeks
    X["average_rides_last_4_weeks"] = X[last_4_weeks_columns].mean(axis=1)

    return X


# FunctionTransformer to add the average rides feature
add_feature_average_rides_last_4_weeks = FunctionTransformer(
    average_rides_last_4_weeks, validate=False
)


# Custom transformer to add temporal features
class TemporalFeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        X_["hour"] = X_["pickup_hour"].dt.hour
        X_["day_of_week"] = X_["pickup_hour"].dt.dayofweek
        X_["is_weekend"] = X_["day_of_week"].isin([5, 6]).astype(int)


        return X_.drop(columns=["pickup_hour", "pickup_location_id"])


# Instantiate the temporal feature engineer
add_temporal_features = TemporalFeatureEngineer()

scalar = StandardScaler()
# Function to return the pipeline
def get_pipeline(**hyper_params):
    """
    Returns a pipeline with optional parameters for LGBMRegressor.

    Parameters:
    ----------
    **hyper_params : dict
        Optional parameters to pass to the LGBMRegressor.

    Returns:
    -------
    pipeline : sklearn.pipeline.Pipeline
        A pipeline with feature engineering and LGBMRegressor.
    """
    pipeline = make_pipeline(
        add_feature_average_rides_last_4_weeks,
        add_temporal_features,
        scalar,
        lgb.LGBMRegressor(**hyper_params),  # Pass optional parameters here
    )
    return pipeline
