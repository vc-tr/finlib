#TODO :run python scripts/analysis/compare_variants.py

from train import run_training
import mlflow

for m in ["lstm","gru","bilstm"]:
    mlflow.set_experiment("variant_comparison")
    with mlflow.start_run(run_name=m):
        mlflow.log_param("model_type", m)
        res = run_training(epochs=3, patience=1, model_type=m)
        mlflow.log_metrics(res)