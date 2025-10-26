# MLP TRAINING (PyTorch)
import torch, os, numpy as np
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, f1_score, classification_report
import joblib

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:", device)

# load preprocessed arrays (from previous block)
# X_train, X_val, X_test, y_train, y_val, y_test are in memory. If in separate script, save & load via joblib/npy.

X_train = np.load('models/X_train.npy')
y_train = np.load('models/y_train.npy')
X_val   = np.load('models/X_val.npy')
y_val   = np.load('models/y_val.npy')
X_test  = np.load('models/X_test.npy')
y_test  = np.load('models/y_test.npy')

# convert to tensors
Xtr = torch.tensor(X_train, dtype=torch.float32)
Xv = torch.tensor(X_val, dtype=torch.float32)
Xt = torch.tensor(X_test, dtype=torch.float32)
ytr = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
yv = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
yt = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

train_ds = TensorDataset(Xtr, ytr)
val_ds = TensorDataset(Xv, yv)
test_ds = TensorDataset(Xt, yt)

batch_size = 1024
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

input_dim = X_train.shape[1]
print("input dim:", input_dim)

# model
class MLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

model = MLP(input_dim).to(device)
# compute pos_weight for BCEWithLogitsLoss
pos = y_train.sum()
neg = len(y_train) - pos
pos_weight = (neg / pos) if pos>0 else 1.0
print("pos, neg, pos_weight:", pos, neg, pos_weight)

criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32).to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

# training loop
best_val_auc = 0.0
patience = 6
stale = 0
for epoch in range(1, 51):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    # evaluate on val
    model.eval()
    preds = []
    ys = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            logits = model(xb).cpu().numpy().ravel()
            preds.append(logits)
            ys.append(yb.numpy().ravel())
    preds = np.concatenate(preds)
    ys = np.concatenate(ys)
    probs = 1 / (1 + np.exp(-preds))
    val_auc = roc_auc_score(ys, probs)
    val_pred_labels = (probs >= 0.5).astype(int)
    val_f1 = f1_score(ys, val_pred_labels)
    scheduler.step(val_auc)
    print(f"Epoch {epoch}: val_auc={val_auc:.4f} val_f1={val_f1:.4f}")
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        stale = 0
        torch.save(model.state_dict(), 'models/best_mlp.pth')
    else:
        stale += 1
    if stale >= patience:
        print("Early stopping")
        break

# final evaluation on test
model.load_state_dict(torch.load('models/best_mlp.pth'))
model.eval()
preds = []
ys = []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        logits = model(xb).cpu().numpy().ravel()
        preds.append(logits); ys.append(yb.numpy().ravel())
preds = np.concatenate(preds); ys = np.concatenate(ys)
probs = 1 / (1 + np.exp(-preds))
test_auc = roc_auc_score(ys, probs)
test_pred = (probs >= 0.5).astype(int)
test_f1 = f1_score(ys, test_pred)
print("TEST AUC:", round(test_auc,4), "TEST F1:", round(test_f1,4))
print(classification_report(ys, test_pred))
