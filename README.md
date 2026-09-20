# Citi Bike demand forecasting, end to end

Predicts how many rides start at each Citi Bike station in New York over the next six hours, and keeps that prediction fresh on a schedule. The point is the pipeline, not the model: a feature store, a model registry, scheduled inference, and a monitoring dashboard, all driven by GitHub Actions.

## Result

| Model | Test MAE (rides / station / 6h) |
|---|---|
| Naive baseline | 139.0 |
| Last four weeks average | 45.3 |
| LightGBM | 34.79 |
| LightGBM, tuned | **34.39** |

Tuning helped less than the feature work did.

## How it works

1. **Feature pipeline** (`pipelines/`, `src/feature_pipeline.py`): pulls monthly trip files from the public S3 bucket, cleans them, aggregates rides per station and six-hour bucket, and writes the result to a Hopsworks feature group.
2. **Training pipeline**: builds lagged demand features, trains LightGBM, logs runs to MLflow via DagsHub, and pushes the winning model to the Hopsworks model registry.
3. **Inference pipeline** (`src/inference.py`): loads the latest features and model, writes predictions back to a feature group.
4. **Monitor** (`frontend/frontend_monitor.py`): a Streamlit app that plots predicted versus actual demand and error by hour.

The three pipelines run as chained workflows in `.github/workflows/`: `feature_pipeline.yaml` triggers `model_training_pipeline.yaml`, which triggers `inference_pipeline.yaml`. They need a `HOPSWORKS_API_KEY` secret.

## Notebooks

`notebooks/01` to `16` walk through the project in order: fetching and validating data, feature engineering, baselines, LightGBM, hyperparameter tuning (`11_lgm_hyper.ipynb`), the feature store, model registry, inference, and monitoring.

## Stack

Python, pandas, LightGBM, scikit-learn, Hopsworks, MLflow, Streamlit, GeoPandas, GitHub Actions.

## Run it

```sh
pip install -r requirements.txt
cp .env.example .env   # add HOPSWORKS_API_KEY
streamlit run frontend/frontend_monitor.py
```

## Honest note

This was built as a course project at the University at Buffalo, Spring 2025, starting from an instructor template for NYC taxi demand. The adaptation to a new dataset, the station-level features, the zone geometry and the monitoring are mine; the overall pipeline shape is the course's.
