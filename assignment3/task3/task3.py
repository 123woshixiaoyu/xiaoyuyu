#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 10:51:35 2024

@author: magnuso
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 3 — Simple CNN on Face/NonFace
- 100 trials, each with a fresh 80/20 split
- report mean train error and mean test error (no ROC)
"""

import os
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- CNN ----------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding='same')
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 3, padding='same')
        self.conv3 = nn.Conv2d(16, 32, 3, padding='same')
        self.bn1   = nn.BatchNorm2d(8)
        self.bn2   = nn.BatchNorm2d(16)
        self.bn3   = nn.BatchNorm2d(32)
        self.fc1   = nn.Linear(4*4*32, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 19->9
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 9->4
        x = F.relu(self.bn3(self.conv3(x)))             # keep 4x4
        x = torch.flatten(x, 1)
        return self.fc1(x)  # logits

# -------------- utils --------------
def load_facenonface():
    candidates = [
        "./FaceNonFace.mat",
        "../FaceNonFace.mat",
        "../../FaceNonFace.mat",
        "../task1/FaceNonFace.mat",
        "../task2/FaceNonFace.mat",
    ]
    for p in candidates:
        if os.path.exists(p):
            return sio.loadmat(p)
    raise FileNotFoundError("FaceNonFace.mat not found near task3.py")

def split_8020(X, Y, split=0.8, seed=None):
    # simple random permutation split (和老师示例一致，不做分层以保持代码简洁)
    if seed is not None:
        torch.manual_seed(seed)
    n = X.size(0)
    idx = torch.randperm(n)
    t = int(split * n)
    return X[idx[:t]], X[idx[t:]], Y[idx[:t]], Y[idx[t:]]

def zscore_by_train(Xtr, Xte):
    # 仅用训练集统计做标准化
    mu = Xtr.mean()
    sd = Xtr.std()
    sd = sd if sd > 1e-8 else torch.tensor(1.0, dtype=Xtr.dtype, device=Xtr.device)
    return (Xtr - mu)/sd, (Xte - mu)/sd

@torch.no_grad()
def accuracy(net, X, Y):
    logits = net(X)
    pred = logits.argmax(dim=1)
    return (pred == Y).float().mean().item()

def train_once(Xtr, Ytr, Xte, Yte, epochs=100, lr=0.01, momentum=0.9, device="cpu", verbose=False):
    net = SimpleCNN().to(device)
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    crit = nn.CrossEntropyLoss()

    Xtr = Xtr.to(device); Ytr = Ytr.to(device)
    Xte = Xte.to(device); Yte = Yte.to(device)

    net.train()
    for ep in range(epochs):
        opt.zero_grad()
        logits = net(Xtr)
        loss = crit(logits, Ytr.long())
        loss.backward()
        opt.step()
        if verbose and ((ep+1) % 10 == 0 or ep == 0):
            print(f"[CNN] Epoch {ep+1:3d}/{epochs} | loss={loss.item():.4f}")

    net.eval()
    acc_tr = accuracy(net, Xtr, Ytr)
    acc_te = accuracy(net, Xte, Yte)
    return 1.0 - acc_tr, 1.0 - acc_te  # return errors

# -------------- main --------------
if __name__ == "__main__":
    # load data
    mat = load_facenonface()
    X = mat["X"].T.astype("float32")              # (N, 361)
    Y = mat["Y"].reshape(-1).astype("int64")      # (-1, +1)
    Y = ((Y + 1) // 2).astype("int64")            # -> {0,1}

    N = X.shape[0]
    X = torch.from_numpy(X).view(N, 1, 19, 19)    # (N,1,19,19)
    Y = torch.from_numpy(Y)                       # (N,)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    n_trials = 100
    train_errs = []
    test_errs  = []

    base_seed = 42  # 固定起点，保证可复现；每个 trial 用不同 seed
    for t in range(n_trials):
        seed = base_seed + t
        Xtr, Xte, Ytr, Yte = split_8020(X, Y, split=0.8, seed=seed)

        # 标准化（仅用训练集统计）
        Xtr_n, Xte_n = zscore_by_train(Xtr, Xte)

        tr_err, te_err = train_once(
            Xtr_n, Ytr, Xte_n, Yte,
            epochs=100, lr=0.01, momentum=0.9, device=device, verbose=False
        )
        train_errs.append(tr_err)
        test_errs.append(te_err)

    train_errs = np.array(train_errs, dtype=float)
    test_errs  = np.array(test_errs, dtype=float)

    print(f"Mean train error over {n_trials} trials: {train_errs.mean():.3f} ± {train_errs.std():.3f}")
    print(f"Mean test  error over {n_trials} trials: {test_errs.mean():.3f} ± {test_errs.std():.3f}")


