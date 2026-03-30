# Bike Demand Prediction Dashboard (Flask + ML)

Production-ready MVP dashboard to upload bike-sharing datasets, preprocess data, train regression models, evaluate performance, visualize analytics, and make predictions.

## Features
- Upload CSV/Excel datasets and preview/summary
- Preprocess (missing values, one-hot encoding, scaling, train/test split)
- Train models: Linear Regression, Decision Tree, Random Forest, Gradient Boosting (+ XGBoost if installed)
- Metrics: R², Adjusted R², MAE, MSE, RMSE
- 10+ charts via Chart.js (and small plugins for heatmap/boxplot)
- Prediction form generated dynamically from model features
- SQLite persistence: datasets, trained models, prediction history
- Export predictions/models as CSV

## Quickstart (local)
```bash
cd bike-demand-dashboard
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000/dashboard`.

## Optional: XGBoost
```bash
pip install xgboost
```
The app automatically enables XGBoost if it imports successfully.

## Example dataset format
Any bike-demand style dataset works as long as:
- It’s a tabular CSV/Excel file
- The **target column** is numeric (e.g., `count`, `demand`, `rides`)

Recommended columns (common in bike sharing):
- `datetime` (or `date`, `timestamp`)
- `temp`, `humidity`, `windspeed`
- `season`, `weather`, `holiday`, `workingday`
- Target: `count` (rides/demand)

An example CSV is included at `example_data/bike_demand_example.csv`.

## Project structure
See folders under `models/`, `routes/`, `services/`, `utils/`, `templates/`, `static/`.

## Pages
- `/dashboard` Dashboard overview
- `/upload` Upload + preview + target selection
- `/preprocess` Preprocessing configuration
- `/train` Model training
- `/performance` Model comparison + residuals
- `/predictions` Dynamic prediction form + history
- `/analytics` 10 charts dashboard
- `/reports` CSV exports
- `/settings` App info

## API routes (core)
- `POST /api/datasets/upload`
- `GET /api/datasets`
- `GET /api/datasets/<id>/preview`
- `GET /api/datasets/<id>/summary`
- `POST /api/preprocess/<dataset_id>`
- `POST /api/train`
- `GET /api/models`
- `GET /api/models/<id>/metrics`
- `POST /api/predict/<model_id>`
- `GET /api/predictions`
- `GET /api/dashboard/stats`
- `GET /api/dashboard/charts/<dataset_id>`
- `GET /api/dashboard/insights/<model_id>`
- `GET /api/export/predictions`
- `GET /api/export/models`
- `GET /api/export/report/<model_id>`

## Future improvements
- Authentication + multi-user projects
- Hyperparameter tuning + cross-validation
- Automated feature importance/SHAP explanations
- Dataset versioning and model registry
- PDF report generation
