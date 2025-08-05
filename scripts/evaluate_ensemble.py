from src.pipeline.dataset import get_dataloader  # assume a helper
from src.models.ensemble import ensemble_average, ensemble_weighted
from sklearn.metrics import mean_squared_error, mean_absolute_error

# specify your model checkpoints
models = ["runs/exp1/best_model", "runs/exp2/best_model"]
weights = [0.5, 0.5]  # or derived from val losses

loader = get_dataloader(split="test")
true = pd.concat([y for _,y in loader], axis=0)

# Average ensemble
pred_avg = ensemble_average(models, loader)
# Weighted ensemble
pred_wt  = ensemble_weighted(models, weights, loader)

for name, pred in [("avg",pred_avg), ("wt",pred_wt)]:
    print(name,
          "MAE:", round(mean_absolute_error(true,pred),4),
          "RMSE:", round(mean_squared_error(true,pred, squared=False),4))