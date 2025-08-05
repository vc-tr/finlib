import time, torch
from train import prepare_data, build_model  # refactor these into functions
def profile(device, num_workers, pin_memory):
    torch.cuda.set_device(device) if device=="cuda" else None
    start = time.time()
    ds_train, _ = prepare_data()
    loader = torch.utils.data.DataLoader(
        ds_train, batch_size=64, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    model = build_model(device=device)
    model.to(device)
    for xb, _ in loader:
        xb = xb.to(device, non_blocking=pin_memory)
        _ = model(xb)
        break  # just one batch
    print(f"{device} | workers={num_workers} pin={pin_memory} | {(time.time()-start):.4f}s")

if __name__=="__main__":
    for dev in ["cpu", "cuda"]:
        for w in [0,2,4]:
            for pin in [False, True]:
                profile(dev, w, pin)