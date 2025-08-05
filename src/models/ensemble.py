import torch
import pandas as pd

def load_model(path, device="cpu"):
    return torch.jit.load(path, map_location=device).eval()

def predict_with_model(model, dataloader, device="cpu"):
    preds = []
    with torch.no_grad():
        for xb, _ in dataloader:
            xb = xb.to(device)
            preds.append(model(xb).cpu().numpy())
    return pd.Series(np.concatenate(preds))

def ensemble_average(model_paths, dataloader, device="cpu"):
    # unweighted average
    preds = [predict_with_model(load_model(p,device), dataloader, device) 
             for p in model_paths]
    return pd.concat(preds, axis=1).mean(axis=1)

def ensemble_weighted(model_paths, weights, dataloader, device="cpu"):
    preds = [predict_with_model(load_model(p,device), dataloader, device)*w 
             for p,w in zip(model_paths, weights)]
    return pd.concat(preds, axis=1).sum(axis=1)