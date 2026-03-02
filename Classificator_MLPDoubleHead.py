# Use the trained autoencoder (frozen) to compress each signal to 4 latent vars.
# Then train two simple MLP classifiers:
#   - one predicts (mua1, mua2) as binned classes
#   - one predicts (mus1, mus2) as binned classes
# Finally do some plots: accuracy vs (T0,R) and confusion matrices.


import os
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from DatasetNpz import npzDataset
from AutoencoderNpz import Autoencoder
import matplotlib.pyplot as plt


# Indices of the 4 parameters
MUA1_IDX = 2   
MUA2_IDX = 3   
MUS1_IDX = 4   
MUS2_IDX = 5   

# Number of bins for each parameter
MUA1_BINS = 10
MUA2_BINS = 10
MUS1_BINS = 11
MUS2_BINS = 11

BIN_MODE = "quantile"  # "linear?"

BATCH_ENCODE = 1024
BATCH_TRAIN  = 512
BATCH_EVAL   = 1024
EPOCHS       = 100
LR           = 5e-3
RNG_MAIN_SEED = 42

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
#print(device)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load autoencoder and freeze it
model_path = os.path.join(BASE_DIR, "Autoencoder", "autoencoder_model")
autoencoder = torch.load(model_path, map_location=device, weights_only=False)
autoencoder.to(device)
autoencoder.eval()

encoder = autoencoder.encoder
for p in encoder.parameters():
    p.requires_grad = False
encoder.eval()

print("Loaded autoencoder. Encoder frozen.")

# Build dataset and encoding everything
tabella_path = os.path.join(BASE_DIR, "..", "ProcessedPoisson", "legenda.txt")
npz_dir = os.path.dirname(tabella_path)
df = pd.read_csv(tabella_path, delim_whitespace=True)

## keep only existing npz files (it crashes otherwise)
df = df[df.iloc[:, 0].apply(lambda x: os.path.exists(os.path.join(npz_dir, f'{x.rsplit('.mat')[0]}.npz')))]

dataset = npzDataset(df, npz_dir)
encode_loader = DataLoader(dataset, batch_size=BATCH_ENCODE, shuffle=False, pin_memory=True, num_workers=8)

print("N samples:", len(dataset))

# Now Z will contain [T0, R, z1, z2, z3, z4] per sample.
# In the following we collect in Z the latent features of the model - those that would be necessary 
# for the autoencoder to decode - and in y the raw targets.
# We do this as we use the latent features as input for the MLP to predict the targets later on.

geom_dim = 2  # First two entries of X: T0 and R

Z_list, y_list = [], []  # y_list shape [B, 4]

with torch.no_grad():
    for Xbatch, Ybatch in encode_loader:
        # Xbatch shape: [B, n_features], with [T0, R, ... rest ...]
        Xbatch = Xbatch.to(device, non_blocking=True)

        Z_latent = encoder(Xbatch)               # [B, latent_dim=4]
        T0R = Ybatch[:, :geom_dim].to(device)              # [B, 2]
        #R = Ybatch[:, 1:geom_dim].to(device)                 # [B, 1]

        Z_full = torch.cat([T0R, Z_latent], dim=1)  # [B, 6] it's the [T0, R, z1..z4]
        #Z_full = torch.cat([R, Z_latent], dim=1)     # [B, 5] it's the [R, z1,..z4]
        Z_list.append(Z_full.detach().cpu())        #store 6D/5D features

        # Extract raw targets for (mua1, mua2, mus1, mus2)
        
        mu_a1  = Ybatch[:, MUA1_IDX ].cpu().numpy()
        mu_a2  = Ybatch[:, MUA2_IDX].cpu().numpy()
        mu_s1  = Ybatch[:, MUS1_IDX ].cpu().numpy()
        mu_s2  = Ybatch[:, MUS2_IDX].cpu().numpy()

        Y4 = np.stack([mu_a1, mu_a2, mu_s1, mu_s2], axis=1)  # [B, 4]
        y_list.append(Y4)

Z = torch.cat(Z_list, dim=0)  # [N, 6] now [T0, R, z1..z4] or [R, z1..z4]
y_raw = torch.tensor(np.concatenate(y_list, axis=0), dtype=torch.float32)  # [N, 4]
print("Encoded:", Z.shape, "Targets y_raw:", y_raw.shape)

# check
print("\n--- Sanity check: Z vs df ---")
print("Z[:,0] (T0)  min/max:", float(Z[:,0].min()), float(Z[:,0].max()))
print("Z[:,1] (R)   min/max:", float(Z[:,1].min()), float(Z[:,1].max()))

t0_raw = df.iloc[:, 1].to_numpy()
r_raw  = df.iloc[:, 2].to_numpy()
print("df T0 min/max:", t0_raw.min(), t0_raw.max())
print("df R  min/max:", r_raw.min(), r_raw.max())
print("--- end check ---\n")

# Binning continuous targets -> class indices
# Here we convert each continuous target (mua1, mua2, mus1, mus2) into discrete class indices for classification later on.

def make_bins(values_np, n_bins, mode="quantile"):
    if mode == "quantile":
        qs = np.linspace(0, 1, n_bins + 1)
        edges = np.unique(np.quantile(values_np, qs))

        if len(edges) - 1 < n_bins:
            edges = np.linspace(values_np.min(), values_np.max(), n_bins + 1)
    else:
        edges = np.linspace(values_np.min(), values_np.max(), n_bins + 1)
    cls = np.digitize(values_np, edges[1:-1], right=False)  # 0..n_bins-1
    return cls.astype(np.int64), edges

y_np = y_raw.numpy()

# check how many bins are actually present in the data (before re-binning)
print("Unique raw values:")
print("mua1:", np.unique(y_np[:, 0]).size, " expected:", MUA1_BINS)
print("mua2:", np.unique(y_np[:, 1]).size, " expected:", MUA2_BINS)
print("mus1:", np.unique(y_np[:, 2]).size, " expected:", MUS1_BINS)
print("mus2:", np.unique(y_np[:, 3]).size, " expected:", MUS2_BINS)

y_mua1_cls,  bins_mua1  = make_bins(y_np[:,0], MUA1_BINS,  BIN_MODE)
y_mua2_cls, bins_mua2 = make_bins(y_np[:,1], MUA2_BINS, BIN_MODE)
y_mus1_cls,  bins_mus1  = make_bins(y_np[:,2], MUS1_BINS,  BIN_MODE)
y_mus2_cls, bins_mus2 = make_bins(y_np[:,3], MUS2_BINS, BIN_MODE)


Y_cls = torch.tensor(
    np.stack([y_mua1_cls, y_mua2_cls, y_mus1_cls, y_mus2_cls], axis=1),
    dtype=torch.long
)

print("Class labels shape:", Y_cls.shape)


# Train/Test split (80/20) and standardise Z using train stats

rng = np.random.default_rng(RNG_MAIN_SEED)
idx = np.arange(Z.shape[0])
rng.shuffle(idx)
N = len(idx)
N_train = int(0.8 * N)
train_idx, test_idx = idx[:N_train], idx[N_train:]

Z_train, Z_test = Z[train_idx], Z[test_idx]
Y_train_cls, Y_test_cls = Y_cls[train_idx], Y_cls[test_idx]

# save raw copies for plotting (columns 0,1 are T0,R in raw units)
Z_train_raw = Z_train.clone()
Z_test_raw  = Z_test.clone()

# --- Standardize Z using TRAIN stats only ---
Z_mean = Z_train.mean(dim=0, keepdim=True)
Z_std  = Z_train.std(dim=0, keepdim=True).clamp_min(1e-8)

Z_train = (Z_train - Z_mean)/Z_std
Z_test = (Z_test - Z_mean)/Z_std

# Split targets into (mua1, mua2) and (mus1, mus2)
Y_train_mua = Y_train_cls[:, :2]  # [N_train, 2]
Y_train_mus = Y_train_cls[:, 2:]  # [N_train, 2]

Y_test_mua  = Y_test_cls[:, :2]   # [N_test, 2]
Y_test_mus  = Y_test_cls[:, 2:]   # [N_test, 2]

# DataLoaders: one for (mua1, mua2), one for (mus1, mus2)

train_loader_mua = DataLoader(TensorDataset(Z_train, Y_train_mua), batch_size=BATCH_TRAIN, shuffle=True)
test_loader_mua = DataLoader(TensorDataset(Z_test, Y_test_mua), batch_size=BATCH_EVAL, shuffle=False)

train_loader_mus = DataLoader(TensorDataset(Z_train, Y_train_mus), batch_size=BATCH_TRAIN, shuffle=True)
test_loader_mus = DataLoader(TensorDataset(Z_test, Y_test_mus), batch_size=BATCH_EVAL, shuffle=False)

test_loader_full = DataLoader(TensorDataset(Z_test, Y_test_cls), batch_size=BATCH_EVAL, shuffle=False)

# Classifier head (shared body and 2 heads)

latent_dim = Z.shape[1]

class MLPClassifier2(nn.Module):
    def __init__(self, in_dim, hidden1=64, hidden2=32, n_bins=(8, 8)):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
        )
        b1, b2 = n_bins
        self.head1 = nn.Linear(hidden2, b1)
        self.head2 = nn.Linear(hidden2, b2)

    def forward(self, x):
        h = self.shared(x)
        return self.head1(h), self.head2(h)

clf_mua = MLPClassifier2(latent_dim,hidden1=64,hidden2=32,n_bins=(MUA1_BINS, MUA2_BINS)).to(device)
clf_mus = MLPClassifier2(latent_dim,hidden1=64,hidden2=32,n_bins=(MUS1_BINS, MUS2_BINS)).to(device)

criterion = nn.CrossEntropyLoss()
opt_mua = torch.optim.Adam(clf_mua.parameters(), lr=LR)
opt_mus = torch.optim.Adam(clf_mus.parameters(), lr=LR)

print("clf_mua:", clf_mua)
print("clf_mus:", clf_mus)


# Evaluation: per-head and mean accuracy

def evaluate_cls2(model, loader):
    model.eval()
    correct = np.zeros(2)
    total   = 0
    with torch.no_grad():
        for Xb, Yb in loader:
            Xb = Xb.to(device)
            Yb = Yb.to(device)
            logits1, logits2 = model(Xb)   # 2 heads
            preds1 = logits1.argmax(dim=1)
            preds2 = logits2.argmax(dim=1)
            correct[0] += (preds1 == Yb[:, 0]).sum().item()
            correct[1] += (preds2 == Yb[:, 1]).sum().item()
            total += Yb.size(0)
    acc = correct / max(1, total)
    return {"acc_1":  float(acc[0]),"acc_2":  float(acc[1]),"acc_mean": float(acc.mean())}

# Training loop + early stopping

print("\n=== Training classifier for (mua1, mua2) ===")

best_state_mua = None
best_acc_mua  = -1.0
best_epoch_mua = -1

for epoch in range(1, EPOCHS + 1):
    clf_mua.train()
    running_loss = 0.0

    for Xb, Yb in train_loader_mua:
        Xb = Xb.to(device)
        Yb = Yb.to(device)

        opt_mua.zero_grad()
        out1, out2 = clf_mua(Xb)  # logits

        loss  = criterion(out1, Yb[:, 0])
        loss += criterion(out2, Yb[:, 1])
        loss = loss / 2.0

        loss.backward()
        opt_mua.step()

        running_loss += float(loss.item()) * Xb.size(0)

    train_loss = running_loss / len(train_loader_mua.dataset)
    train_metrics = evaluate_cls2(clf_mua, train_loader_mua)

    print(
        f"[MUA] Epoch {epoch:03d} | Train Loss: {train_loss:.4e} | "
        f"Train acc_mean: {train_metrics['acc_mean']:.4f} "
        f"(mua1 {train_metrics['acc_1']:.3f}, "
        f"mua2 {train_metrics['acc_2']:.3f})"
    )

    # track best on test
    test_metrics_epoch = evaluate_cls2(clf_mua, test_loader_mua)
    if test_metrics_epoch["acc_mean"] > best_acc_mua:
        best_acc_mua = test_metrics_epoch["acc_mean"]
        best_state_mua = clf_mua.state_dict()
        best_epoch_mua = epoch
        print(f"  --> [MUA] New best test acc_mean={best_acc_mua:.4f} at epoch {best_epoch_mua}")

if best_state_mua is not None:
    clf_mua.load_state_dict(best_state_mua)
    print(f"[MUA] Loaded best model from epoch {best_epoch_mua} with test acc_mean={best_acc_mua:.4f}")
else:
    print("[MUA] Warning: best_state_mua is not found, using last-epoch model.")

final_mua = evaluate_cls2(clf_mua, test_loader_mua)
print("FINAL TEST ACCURACIES (MUA):",
      f"mean={final_mua['acc_mean']:.4f} | "
      f"mua1={final_mua['acc_1']:.4f}, "
      f"mua2={final_mua['acc_2']:.4f}")


print("\n=== Training classifier for (mus1, mus2) ===")

best_state_mus = None
best_acc_mus   = -1.0
best_epoch_mus = -1

for epoch in range(1, EPOCHS + 1):
    clf_mus.train()
    running_loss = 0.0

    for Xb, Yb in train_loader_mus:
        Xb = Xb.to(device)
        Yb = Yb.to(device)

        opt_mus.zero_grad()
        out1, out2 = clf_mus(Xb)  # logits

        loss  = criterion(out1, Yb[:, 0])
        loss += criterion(out2, Yb[:, 1])
        loss = loss / 2.0

        loss.backward()
        opt_mus.step()

        running_loss += float(loss.item()) * Xb.size(0)

    train_loss = running_loss / len(train_loader_mus.dataset)
    train_metrics = evaluate_cls2(clf_mus, train_loader_mus)

    print(
        f"[MUS] Epoch {epoch:03d} | Train Loss: {train_loss:.4e} | "
        f"Train acc_mean: {train_metrics['acc_mean']:.4f} "
        f"(mus1 {train_metrics['acc_1']:.3f}, "
        f"mus2 {train_metrics['acc_2']:.3f})"
    )

    # track best on test
    test_metrics_epoch = evaluate_cls2(clf_mus, test_loader_mus)
    if test_metrics_epoch["acc_mean"] > best_acc_mus:
        best_acc_mus = test_metrics_epoch["acc_mean"]
        best_state_mus = clf_mus.state_dict()
        best_epoch_mus = epoch
        print(f"  --> [MUS] New best test acc_mean={best_acc_mus:.4f} at epoch {best_epoch_mus}")

if best_state_mus is not None:
    clf_mus.load_state_dict(best_state_mus)
    print(f"[MUS] Loaded best model from epoch {best_epoch_mus} with test acc_mean={best_acc_mus:.4f}")
else:
    print("[MUS] Warning: best_state_mus is not found, using last-epoch model.")

final_mus = evaluate_cls2(clf_mus, test_loader_mus)
print("FINAL TEST ACCURACIES (MUS):",
      f"mean={final_mus['acc_mean']:.4f} | "
      f"mus1={final_mus['acc_1']:.4f}, "
      f"mus2={final_mus['acc_2']:.4f}")


# Final TEST evaluation (2 nets)

# Load BEST model for (mua1, mua2)
if best_state_mua is not None:
    clf_mua.load_state_dict(best_state_mua)
    print(f"[MUA] Loaded best model from epoch {best_epoch_mua} with test acc_mean={best_acc_mua:.4f}")
else:
    print("[MUA] Warning: best_state_mua is None, using last-epoch model.")

final_mua = evaluate_cls2(clf_mua, test_loader_mua)
print("FINAL TEST ACCURACIES (MUA):",
      f"mean={final_mua['acc_mean']:.4f} | "
      f"mua1={final_mua['acc_1']:.4f}, "
      f"mua2={final_mua['acc_2']:.4f}")

# Load BEST model for (mus1, mus2)
if best_state_mus is not None:
    clf_mus.load_state_dict(best_state_mus)
    print(f"[MUS] Loaded best model from epoch {best_epoch_mus} with test acc_mean={best_acc_mus:.4f}")
else:
    print("[MUS] Warning: best_state_mus is None, using last-epoch model.")

final_mus = evaluate_cls2(clf_mus, test_loader_mus)
print("FINAL TEST ACCURACIES (MUS):",
      f"mean={final_mus['acc_mean']:.4f} | "
      f"mus1={final_mus['acc_1']:.4f}, "
      f"mus2={final_mus['acc_2']:.4f}")


# (Combined) 4-head view for analysis

# Wrapping the two 2-head classifiers into a one 4-head classifier
class CombinedClassifier(nn.Module):
    def __init__(self, clf_mua, clf_mus):
        super().__init__()
        self.clf_mua = clf_mua
        self.clf_mus = clf_mus

    def forward(self, x):
        out_mua1, out_mua2 = self.clf_mua(x)
        out_mus1, out_mus2 = self.clf_mus(x)
        return (out_mua1, out_mua2, out_mus1, out_mus2)

clf_combined = CombinedClassifier(clf_mua, clf_mus).to(device)

# collect test predictions
def get_preds_and_targets(model, loader):
    model.eval()
    y_true = [[] for _ in range(4)]
    y_pred = [[] for _ in range(4)]
    with torch.no_grad():
        for Xb, Yb in loader:
            Xb = Xb.to(device)
            Yb = Yb.to(device)
            logits = model(Xb)                          # 4 tensors [B, n_bins_i]
            preds  = [l.argmax(dim=1) for l in logits]  # 4 tensors [B]
            for i in range(4):
                y_true[i].append(Yb[:, i].cpu().numpy())
                y_pred[i].append(preds[i].cpu().numpy())
    return [np.concatenate(y_true[i]) for i in range(4)], [np.concatenate(y_pred[i]) for i in range(4)]


y_true, y_pred = get_preds_and_targets(clf_combined, test_loader_full)
names = ['mua1', 'mua2', 'mus1', 'mus2']
n_bins_each = [MUA1_BINS, MUA2_BINS, MUS1_BINS, MUS2_BINS]

# Accuracy as a function of T0 and R (geometrical parameters)

def accuracy_vs_T0_R_fixed(model, loader, device, t0_raw, r_raw, t0_values, r_values):
    model.eval()
    corr_lists = [[] for _ in range(4)]

    # Collect correctness per head in loader order
    with torch.no_grad():
        for Xb, Yb in loader:
            Xb = Xb.to(device)
            Yb = Yb.to(device)
            logits = model(Xb)
            preds  = [l.argmax(dim=1) for l in logits]
            for i in range(4):
                corr = (preds[i] == Yb[:, i]).float().cpu().numpy()
                corr_lists[i].append(corr)

    corr_all = [np.concatenate(corr_lists[i]) for i in range(4)]

    # Map raw T0/R to categorical indices
    t0_axis = list(t0_values)
    r_axis  = list(r_values)
    t0_map  = {round(v, 6): i for i, v in enumerate(t0_axis)}
    r_map   = {round(v, 6): j for j, v in enumerate(r_axis)}

    t0_idx = np.array([t0_map.get(round(v, 6), -1) for v in t0_raw], dtype=int)
    r_idx  = np.array([r_map.get(round(v, 6), -1) for v in r_raw], dtype=int)

    # Keep only samples that match known categories
    mask = (t0_idx >= 0) & (r_idx >= 0)
    if not np.any(mask):
        raise RuntimeError("No samples matched the provided T0/R value sets.")

    t0_idx = t0_idx[mask]
    r_idx  = r_idx[mask]
    flat   = t0_idx * len(r_axis) + r_idx
    maxf   = len(t0_axis) * len(r_axis)

    mats = []
    for h in range(4):
        corr = corr_all[h][mask]
        sums   = np.bincount(flat, weights=corr, minlength=maxf)
        counts = np.bincount(flat, minlength=maxf)
        with np.errstate(divide='ignore', invalid='ignore'):
            acc = sums / counts
        acc = acc.reshape(len(t0_axis), len(r_axis))
        acc = np.where(counts.reshape(len(t0_axis), len(r_axis)) > 0, acc, np.nan)
        mats.append(acc)

    stack = np.stack(mats, axis=0)     # [4, T0, R]
    valid = ~np.isnan(stack)
    num   = np.nansum(stack, axis=0)
    den   = valid.sum(axis=0)
    mean_mat = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)

    return np.array(t0_axis, dtype=float), np.array(r_axis, dtype=float), mean_mat, mats, True


# Known discrete sets (sorted)
T0_VALUES = [2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
R_VALUES  = [10.0, 20.0, 30.0]

# Raw T0/R for the test set
t0_raw_test = Z_test_raw[:, 0].cpu().numpy()
r_raw_test  = Z_test_raw[:, 1].cpu().numpy()

t0_vals, r_vals, mean_mat, mats, has_t0 = accuracy_vs_T0_R_fixed(
    clf_combined, test_loader_full, device,
    t0_raw=t0_raw_test, r_raw=r_raw_test,
    t0_values=T0_VALUES, r_values=R_VALUES
)

mat_names = ["mean", "mua1", "mua2", "mus1", "mus2"]
mat_list  = [mean_mat] + mats

for k, mat in enumerate(mat_list):
    plt.figure(figsize=(8, 6))
    im = plt.imshow(mat, origin='upper', aspect='auto', vmin=0.0, vmax=1.0, cmap="viridis")

    plt.xticks(ticks=np.arange(len(r_vals)),labels=[f"{v:g}" for v in r_vals],rotation=0)
    plt.yticks(ticks=np.arange(len(t0_vals)),labels=[f"{v:g}" for v in t0_vals])

    plt.xlabel("R")
    plt.ylabel("T0")
    plt.title(f"Accuracy vs (T0, R) - {mat_names[k]}")
    cbar = plt.colorbar(im)
    cbar.set_label("Accuracy")

    # Overlay numeric accuracy values
    n_t0, n_r = mat.shape
    for i in range(n_t0):
        for j in range(n_r):
            val = mat[i, j]
            if not np.isnan(val):
                plt.text(j, i, f"{val:.2f}",
                    ha="center", va="center",
                    color="white" if val < 0.5 else "black",
                    fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(f"Accuracy2D_{mat_names[k]}.png", dpi=300, bbox_inches="tight")
    plt.close()

# confusion matrices
def confusion_matrix_np(y_t, y_p, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_t, y_p):
        cm[t, p] += 1
    return cm

cms = [confusion_matrix_np(y_true[i], y_pred[i], n_bins_each[i]) for i in range(4)]
row_norm = [cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None) for cm in cms]
accs = [np.trace(cm)/cm.sum() if cm.sum()>0 else 0.0 for cm in cms]

# single figure with 4 panels
fig, axes = plt.subplots(2, 2, figsize=(10, 9))
axes = axes.ravel()
vmin, vmax = 0.0, 1.0
im = None

for i, ax in enumerate(axes):
    m = row_norm[i]
    im = ax.imshow(m, vmin=vmin, vmax=vmax, origin='upper', aspect='auto')
    ax.set_title(f"{names[i]}  (acc={accs[i]:.3f})")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks(range(m.shape[1])); ax.set_yticks(range(m.shape[0]))
    for r in range(m.shape[0]):
        for c in range(m.shape[1]):
            ax.text(c, r, f"{m[r, c]:.2f}", ha="center", va="center", fontsize=7)

# one shared colorbar
fig.colorbar(im, ax=axes.tolist(), fraction=0.03, pad=0.02, label="Frequency")
fig.suptitle("Confusion matrices (row-normalized)", y=0.995)
fig.savefig("ConfusionMatrix.png", dpi=300, bbox_inches="tight")
plt.show()