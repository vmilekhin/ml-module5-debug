# train.py — полный демонстрационный код модуля 5
import argparse
import json
import logging
import random
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

class ForecastLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 32, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(1)

def make_windows(a, n=20):
    x = torch.stack([a[i:i+n] for i in range(len(a)-n)])
    return x.unsqueeze(-1), a[n:]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fault", choices=["none", "no-step", "bad-labels", "high-lr"], default="none")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    
    t = np.linspace(0, 80, 4000)
    series = torch.tensor(np.sin(t)+.05*rng.normal(size=len(t)), dtype=torch.float32)
    X, y = make_windows(series)
    cut = int(len(X)*.8)
    X_train, y_train = X[:cut], y[:cut]
    X_test, y_test = X[cut:], y[cut:]
    
    if args.fault == "bad-labels":
        y_train = y_train[torch.randperm(len(y_train))]
        
    loader = DataLoader(TensorDataset(X_train,y_train), batch_size=64, shuffle=True)
    model = ForecastLSTM()
    criterion = nn.MSELoss()
    lr = 1.0 if args.fault == "high-lr" else .001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    writer = SummaryWriter("runs/" + args.fault)
    start = time.perf_counter()
    
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        grad_norm = 0.0
        delta_w = 0.0
        for step, (xb, yb) in enumerate(loader):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(xb)
            loss = criterion(prediction, yb)
            if args.debug and epoch == 0 and step == 0:
                breakpoint()
            loss.backward()
            grad_norm = model.fc.weight.grad.norm().item()
            before = model.fc.weight.detach().clone()
            if args.fault != "no-step":
                optimizer.step()
            delta_w = (model.fc.weight.detach()-before).norm().item()
            total += loss.item()*len(xb)
            
        model.eval()
        with torch.inference_mode():
            test_mse = criterion(model(X_test), y_test).item()
        train_mse = total/len(X_train)
        
        writer.add_scalar("Loss/train", train_mse, epoch)
        writer.add_scalar("Loss/test", test_mse, epoch)
        writer.add_scalar("Diagnostics/grad_norm", grad_norm, epoch)
        writer.add_scalar("Diagnostics/delta_weight", delta_w, epoch)
        writer.add_histogram("Weights/fc", model.fc.weight, epoch)
        
        logging.info(json.dumps({
            "epoch": epoch+1, "train_mse": round(train_mse,6),
            "test_mse": round(test_mse,6), "grad_norm": round(grad_norm,6),
            "delta_weight": round(delta_w,6), "lr": lr
        }))
        
    baseline_mse = criterion(X_test[:, -1, 0], y_test).item()
    print("Baseline MSE:", baseline_mse)
    print("Model TEST MSE:", test_mse)
    print("Elapsed seconds:", round(time.perf_counter()-start,2))
    torch.save({
        "epoch": args.epochs, "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(), "lr": lr,
        "fault": args.fault
    }, "checkpoint.pth")
    writer.close()

if __name__ == "__main__":
    main()
