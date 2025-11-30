# src/models/train.py
import os
import yaml
import json
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import compute_sample_weight

# Optional model imports are inside functions to avoid hard deps at import time

# ---------- CONFIG ----------
PARAMS = "params.yaml"
PROCESSED = "data/processed/city_day_processed.csv"
MODEL_OUT = "model.joblib"
MLFLOW_EXPERIMENT = "india_aqi_classification"
ARTIFACT_DIR = "artifacts"

# ---------- UTILITIES ----------
def load_params(path=PARAMS):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_data(path=PROCESSED):
    if not os.path.exists(path):
        raise FileNotFoundError("Processed data not found. Run preprocessing first.")
    return pd.read_csv(path)

def set_seed(seed: int):
    np.random.seed(seed)

def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def plot_and_save_confusion_matrix(y_true, y_pred, classes, out_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           ylabel='True label',
           xlabel='Predicted label',
           title='Confusion Matrix')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

# ---------- MODEL TRAINING FUNCTIONS ---------- #

def rf_random_search(X, y, params, cv_splits=3, n_iter=25, seed=42):
    rf_base_params = params.get('model', {}).get('rf', {})
    clf = RandomForestClassifier(random_state=seed, n_jobs=-1)

    # search space (some sensible ranges, but will respect values in params if provided)
    from scipy.stats import randint, uniform
    param_dist = {
        'n_estimators': randint(100, 600),
        'max_depth': randint(4, 30),
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 6),
        'criterion': ['gini', 'entropy']
    }

    # override defaults if user set them explicitly
    for k, v in rf_base_params.items():
        if k in ['n_estimators', 'max_depth']:
            # if set fixed in params, lock those values by setting a tight distribution
            param_dist[k] = [v]

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    rs = RandomizedSearchCV(clf, param_distributions=param_dist,
                            n_iter=n_iter, cv=cv, scoring='accuracy', verbose=1,
                            random_state=seed, n_jobs=-1)
    rs.fit(X, y)
    return rs.best_estimator_, rs.best_score_, rs.cv_results_

def train_lightgbm(X_train, y_train, X_val, y_val, params, use_gpu=False, seed=42):
    import lightgbm as lgb
    gbm_p = params['model']['gbm'].copy()
    # align defaults with params
    n_estimators = gbm_p.pop('n_estimators', 1000)
    num_class = int(np.unique(y_train).shape[0])

    lgb_params = {
        'objective': 'multiclass',
        'num_class': num_class,
        'metric': 'multi_logloss',
        'verbosity': -1,
        'seed': seed,
        **gbm_p
    }
    if use_gpu:
        lgb_params.update({'device': 'gpu', 'gpu_platform_id': 0, 'gpu_device_id': 0})

    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    bst = lgb.train(lgb_params, train_set=dtrain,
                    num_boost_round=n_estimators,
                    valid_sets=[dvalid],
                    early_stopping_rounds=50,
                    verbose_eval=False)

    preds_proba = bst.predict(X_val, num_iteration=bst.best_iteration)
    preds = np.argmax(preds_proba, axis=1)
    acc = accuracy_score(y_val, preds)
    return bst, acc

def train_xgboost(X_train, y_train, X_val, y_val, params, use_gpu=False, seed=42):
    import xgboost as xgb
    gbm_p = params['model']['gbm'].copy()
    n_estimators = gbm_p.pop('n_estimators', 1000)
    num_class = int(np.unique(y_train).shape[0])

    xgb_params = {
        'max_depth': gbm_p.get('max_depth', 6),
        'eta': gbm_p.get('learning_rate', 0.1),
        'objective': 'multi:softprob',
        'num_class': num_class,
        'verbosity': 0,
        'seed': seed
    }
    if use_gpu:
        xgb_params.update({'tree_method': 'gpu_hist', 'predictor': 'gpu_predictor'})

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_val, label=y_val)
    bst = xgb.train(xgb_params, dtrain, num_boost_round=n_estimators,
                    evals=[(dvalid, 'valid')],
                    early_stopping_rounds=50,
                    verbose_eval=False)
    preds_proba = bst.predict(dvalid, ntree_limit=bst.best_ntree_limit)
    preds = np.argmax(preds_proba, axis=1)
    acc = accuracy_score(y_val, preds)
    return bst, acc

# ---------- MAIN ----------
def main():
    params = load_params()
    seed = params.get('train', {}).get('random_state', 42)
    set_seed(seed)

    df = load_data()
    target = params.get('target_col', 'AQI_Bucket')
    if 'AQI_Bucket_label' in df.columns:
        target = 'AQI_Bucket_label'
    if target not in df.columns:
        raise ValueError(f"Target column {target} missing. Available: {df.columns.tolist()}")

    print(f"Using target column: {target}")
    # clean target
    df = df.dropna(subset=[target]).copy()
    df[target] = pd.to_numeric(df[target], errors='coerce')
    df = df.dropna(subset=[target])
    df[target] = df[target].astype(int)

    # drop any extraneous non-feature columns if present
    # We'll keep everything except target and AQI numeric
    X = df.drop(columns=[target, 'AQI'], errors='ignore')
    y = df[target].values

    print("Dataset shape:", X.shape, "Num classes:", len(np.unique(y)))

    # separate numeric vs other columns (processed dataset likely mostly numeric)
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    other_cols = [c for c in X.columns if c not in numeric_cols]

    # build preprocessing pipeline
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    # If there are still non-numeric columns (rare because preprocess created dummies), convert with simple imputer + scaler
    other_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("other", other_transformer, other_cols)
        ],
        remainder='drop'  # drop any columns not explicitly listed
    )

    # train / test split (stratified)
    test_size = params.get('train', {}).get('test_size', 0.2)
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed)

    # fit preprocessor on training data and transform
    X_train = preprocessor.fit_transform(X_train_df)
    X_test = preprocessor.transform(X_test_df)

    # prepare MLflow
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run():
        # log params top-level
        mlflow.log_params({
            "seed": seed,
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "n_classes": int(len(np.unique(y)))
        })
        mlflow.log_params(params.get("model", {}))

        use_gpu = params.get('use_gpu', False)
        gpu_backend = params.get('gpu_backend', 'lightgbm')

        best_model = None
        best_acc = 0.0
        best_report = None

        # Option: class weights if imbalance (helps some datasets)
        classes, counts = np.unique(y_train, return_counts=True)
        imbalance_ratio = counts.max() / counts.min() if counts.min() > 0 else 1.0
        print("Class counts:", dict(zip(classes, counts)), "imbalance_ratio:", imbalance_ratio)
        if imbalance_ratio > 3:
            print("Large imbalance detected — will compute sample weights.")
            sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
        else:
            sample_weights = None

        # ---------- GPU PATH ----------
        if use_gpu and gpu_backend in ('lightgbm', 'xgboost'):
            print("Trying GPU training with", gpu_backend)
            try:
                if gpu_backend == 'lightgbm':
                    model_obj, acc = train_lightgbm(X_train, y_train, X_test, y_test, params, use_gpu=True, seed=seed)
                    best_model, best_acc = model_obj, acc
                    mlflow.log_param("gpu_backend", "lightgbm")
                elif gpu_backend == 'xgboost':
                    model_obj, acc = train_xgboost(X_train, y_train, X_test, y_test, params, use_gpu=True, seed=seed)
                    best_model, best_acc = model_obj, acc
                    mlflow.log_param("gpu_backend", "xgboost")
            except Exception as e:
                print("GPU training failed; falling back to RF. Error:", e)
                mlflow.log_param("gpu_fallback", True)
                # fall through to RF training below

        # ---------- CPU baseline + tuning (RandomForest) ----------
        if best_model is None:
            print("Running RandomForest RandomizedSearchCV (CPU)")
            rf_iters = params.get('model', {}).get('rf', {}).get('random_search_iters', 25)
            rf_cv = params.get('model', {}).get('rf', {}).get('cv_splits', 3)
            best_rf, best_cv_score, cv_results = rf_random_search(X_train, y_train, params, cv_splits=rf_cv, n_iter=rf_iters, seed=seed)
            print("Best RF CV score:", best_cv_score)
            best_model = best_rf
            best_acc = None  # will compute below

        # If we used a sklearn estimator, we may want a pipeline that includes preprocessor -> model
        # For sklearn RandomForest we save a sklearn pipeline to ensure transforms persist
        if isinstance(best_model, RandomForestClassifier):
            pipeline = Pipeline([("preprocessor", preprocessor), ("model", best_model)])
            # evaluate on test set
            preds = pipeline.predict(X_test_df)
            acc = accuracy_score(y_test, preds)
            best_acc = acc
            report = classification_report(y_test, preds, output_dict=True)
            best_report = report
            print("Test accuracy (RF pipeline):", acc)
            # log model with mlflow
            mlflow.sklearn.log_model(pipeline, "sklearn-model")
            joblib.dump(pipeline, MODEL_OUT)
            mlflow.log_artifact(MODEL_OUT)
        else:
            # For LightGBM / XGBoost boosters, they expect numpy arrays already transformed
            # Best_model likely is booster object; handle saving accordingly
            try:
                if 'lightgbm' in str(type(best_model)).lower():
                    # best_model is a Booster
                    preds_proba = best_model.predict(X_test, num_iteration=best_model.best_iteration)
                    preds = np.argmax(preds_proba, axis=1)
                    acc = accuracy_score(y_test, preds)
                    best_acc = acc
                    best_report = classification_report(y_test, preds, output_dict=True)
                    save_path = "model_lightgbm.txt"
                    best_model.save_model(save_path)
                    mlflow.log_artifact(save_path)
                else:
                    # xgboost Booster
                    import xgboost as xgb
                    if isinstance(best_model, xgb.core.Booster):
                        dtest = xgb.DMatrix(X_test)
                        preds_proba = best_model.predict(dtest, ntree_limit=getattr(best_model, "best_ntree_limit", None))
                        preds = np.argmax(preds_proba, axis=1)
                        acc = accuracy_score(y_test, preds)
                        best_acc = acc
                        best_report = classification_report(y_test, preds, output_dict=True)
                        save_path = "model.xgb"
                        best_model.save_model(save_path)
                        mlflow.log_artifact(save_path)
                    else:
                        # Unknown model type; attempt joblib
                        try:
                            preds = best_model.predict(X_test)
                            best_acc = accuracy_score(y_test, preds)
                            best_report = classification_report(y_test, preds, output_dict=True)
                            joblib.dump(best_model, MODEL_OUT)
                            mlflow.log_artifact(MODEL_OUT)
                        except Exception as e:
                            print("Couldn't evaluate/save best_model automatically:", e)
            except Exception as e:
                print("Error evaluating non-sklearn model:", e)

        # ---------- LOG METRICS & ARTIFACTS ----------
        if best_acc is not None:
            print("Final test accuracy:", best_acc)
            mlflow.log_metric("test_accuracy", float(best_acc))
        if best_report:
            save_json(best_report, os.path.join(ARTIFACT_DIR, "classification_report.json"))
            mlflow.log_artifact(os.path.join(ARTIFACT_DIR, "classification_report.json"))

        # Confusion matrix plot
        try:
            # ensure preds available
            if 'preds' in locals():
                classes = [str(c) for c in np.unique(y_test)]
                cm_path = os.path.join(ARTIFACT_DIR, "confusion_matrix.png")
                plot_and_save_confusion_matrix(y_test, preds, classes, cm_path)
                mlflow.log_artifact(cm_path)
        except Exception as e:
            print("Confusion matrix plot failed:", e)

    print("Training complete. MLflow run recorded.")
    print("Test accuracy:", best_acc)

if __name__ == "__main__":
    main()