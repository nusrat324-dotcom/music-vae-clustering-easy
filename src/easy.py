import os
import json
import pickle
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)
from sklearn.manifold import TSNE

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# 1. CONFIG
# ============================================================

DATA_DIR = r"C:\Users\mshakur\OneDrive - Oklahoma A and M System\Deep learning\NNN\modelling\Data"
OUTPUT_DIR = r"C:\Users\mshakur\OneDrive - Oklahoma A and M System\Deep learning\NNN\modelling\outputs\easy\final_report_model"

os.makedirs(OUTPUT_DIR, exist_ok=True)

PREFERRED_FILES = [
    "features_3_sec.csv",
    "features_3_sec.xlsx",
    "features_3_sec.xls",
    "features_3_sec.excel",
    "features_30_sec.csv",
    "features_30_sec.xlsx",
    "features_30_sec.xls",
    "features_30_sec.excel"
]

RANDOM_STATE = 42
N_CLUSTERS = 10
BATCH_SIZE = 256

# Final chosen model from ranked results
LATENT_DIM = 10
HIDDEN_DIMS = [128, 64]
BETA_MAX = 0.01
LEARNING_RATE = 5e-4
EPOCHS = 10
WARMUP_EPOCHS = 3
USE_BATCHNORM = False
DENOISE_STD = 0.0

PCA_LATENT_DIM = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

print("=" * 90)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
print("Using device:", DEVICE)
print("=" * 90)


# ============================================================
# 2. HELPERS
# ============================================================

def save_json(obj, filepath):
    with open(filepath, "w") as f:
        json.dump(obj, f, indent=4)

def save_pickle(obj, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)

def find_feature_file(data_dir, preferred_files):
    for fname in preferred_files:
        fpath = os.path.join(data_dir, fname)
        if os.path.exists(fpath):
            return fpath
    raise FileNotFoundError("Could not find GTZAN feature file.")

def load_feature_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(file_path)
    elif ext == ".csv":
        return pd.read_csv(file_path)
    else:
        try:
            return pd.read_excel(file_path)
        except Exception:
            return pd.read_csv(file_path)

def get_beta(epoch, warmup_epochs, beta_max):
    if warmup_epochs <= 0:
        return beta_max
    progress = min(1.0, (epoch + 1) / warmup_epochs)
    return beta_max * progress

def compute_metrics(X_embed, pred_labels, true_labels):
    out = {}
    out["silhouette"] = silhouette_score(X_embed, pred_labels) if len(np.unique(pred_labels)) > 1 else np.nan
    out["calinski_harabasz"] = calinski_harabasz_score(X_embed, pred_labels) if len(np.unique(pred_labels)) > 1 else np.nan
    out["ari"] = adjusted_rand_score(true_labels, pred_labels)
    out["nmi"] = normalized_mutual_info_score(true_labels, pred_labels)
    out["n_clusters_found"] = len(np.unique(pred_labels))
    return out

def plot_training_curves(total_losses, recon_losses, kl_losses, betas, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(total_losses, label="Total Loss")
    plt.plot(recon_losses, label="Reconstruction Loss")
    plt.plot(kl_losses, label="KL Loss")
    plt.plot(betas, label="Beta", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("VAE Training Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_2d_embedding(embedding, labels, title, save_path, class_names=None):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=labels,
        cmap="tab10",
        s=10,
        alpha=0.75
    )
    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.grid(True, alpha=0.25)

    if class_names is not None:
        handles, _ = scatter.legend_elements()
        plt.legend(
            handles,
            class_names,
            title="Classes",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            fontsize=8
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_cluster_vs_genre_heatmap(ctab_df, save_path, title):
    arr = ctab_df.values

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(arr, aspect="auto")

    ax.set_xticks(np.arange(ctab_df.shape[1]))
    ax.set_yticks(np.arange(ctab_df.shape[0]))
    ax.set_xticklabels(ctab_df.columns, rotation=45, ha="right")
    ax.set_yticklabels(ctab_df.index)

    ax.set_xlabel("True Genre")
    ax.set_ylabel("Cluster")
    ax.set_title(title)

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, int(arr[i, j]), ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_metric_bars(results_df, save_path):
    methods = results_df["Method"].tolist()
    x = np.arange(len(methods))
    width = 0.2

    plt.figure(figsize=(12, 6))
    plt.bar(x - 1.5 * width, results_df["Silhouette"], width=width, label="Silhouette")
    plt.bar(x - 0.5 * width, results_df["Calinski_Harabasz"], width=width, label="CH")
    plt.bar(x + 0.5 * width, results_df["ARI"], width=width, label="ARI")
    plt.bar(x + 1.5 * width, results_df["NMI"], width=width, label="NMI")

    plt.xticks(x, methods, rotation=20, ha="right")
    plt.ylabel("Metric Value")
    plt.title("Method Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# 3. LOAD AND PREPROCESS DATA
# ============================================================

feature_file = find_feature_file(DATA_DIR, PREFERRED_FILES)
print(f"Using feature file: {feature_file}")

df = load_feature_file(feature_file)
print("Loaded shape:", df.shape)

df = df.loc[:, ~df.columns.astype(str).str.contains("^Unnamed")]

label_col = None
for candidate in ["label", "genre", "class"]:
    if candidate in df.columns:
        label_col = candidate
        break

if label_col is None:
    raise ValueError("Could not find label column.")

y_text = df[label_col].astype(str).values

drop_cols = [label_col]
if "filename" in df.columns:
    drop_cols.append("filename")
if "length" in df.columns:
    drop_cols.append("length")

X_df = df.drop(columns=drop_cols, errors="ignore")
X_df = X_df.select_dtypes(include=[np.number])

# Robust clipping
lower = X_df.quantile(0.01)
upper = X_df.quantile(0.99)
X_df = X_df.clip(lower=lower, upper=upper, axis=1)

X_df = X_df.fillna(X_df.mean())
X = X_df.values.astype(np.float32)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_text)

pd.DataFrame({
    "encoded_label": np.arange(len(label_encoder.classes_)),
    "class_name": label_encoder.classes_
}).to_csv(os.path.join(OUTPUT_DIR, "label_mapping.csv"), index=False)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X).astype(np.float32)
save_pickle(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))

X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.long)
)
val_dataset = TensorDataset(
    torch.tensor(X_val, dtype=torch.float32),
    torch.tensor(y_val, dtype=torch.long)
)

pin_memory_flag = torch.cuda.is_available()

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=pin_memory_flag
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=pin_memory_flag
)

print("Feature matrix shape:", X.shape)
print("Train shape:", X_train.shape)
print("Val shape:", X_val.shape)
print("Classes:", list(label_encoder.classes_))


# ============================================================
# 4. MODEL
# ============================================================

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, use_batchnorm=False):
        super().__init__()

        encoder_layers = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hdim))
            if use_batchnorm:
                encoder_layers.append(nn.BatchNorm1d(hdim))
            encoder_layers.append(nn.LeakyReLU(0.2))
            prev_dim = hdim
        self.encoder = nn.Sequential(*encoder_layers)

        self.mu_layer = nn.Linear(prev_dim, latent_dim)
        self.logvar_layer = nn.Linear(prev_dim, latent_dim)

        decoder_layers = []
        prev_dim = latent_dim
        for hdim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, hdim))
            if use_batchnorm:
                decoder_layers.append(nn.BatchNorm1d(hdim))
            decoder_layers.append(nn.LeakyReLU(0.2))
            prev_dim = hdim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

def vae_loss_function(recon_x, target_x, mu, logvar, beta):
    recon_loss = nn.MSELoss(reduction="sum")(recon_x, target_x)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss

def extract_latent_mu(model, X_array, device, batch_size=1024):
    model.eval()
    ds = TensorDataset(torch.tensor(X_array, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, pin_memory=pin_memory_flag)

    outputs = []
    with torch.no_grad():
        for (xb,) in tqdm(dl, desc="Extracting latent features", leave=False):
            xb = xb.to(device, non_blocking=True)
            mu, _ = model.encode(xb)
            outputs.append(mu.cpu().numpy())
    return np.vstack(outputs)


# ============================================================
# 5. TRAIN FINAL MODEL
# ============================================================

model = VAE(
    input_dim=X_train.shape[1],
    hidden_dims=HIDDEN_DIMS,
    latent_dim=LATENT_DIM,
    use_batchnorm=USE_BATCHNORM
).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

with open(os.path.join(OUTPUT_DIR, "model_architecture.txt"), "w") as f:
    f.write(str(model))

train_losses = []
train_recons = []
train_kls = []
beta_history = []
epoch_validation_rows = []

print("\nTraining final report model...")

for epoch in range(EPOCHS):
    model.train()
    beta = get_beta(epoch, WARMUP_EPOCHS, BETA_MAX)
    beta_history.append(beta)

    epoch_total = 0.0
    epoch_recon = 0.0
    epoch_kl = 0.0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)

    for batch_x, _ in progress_bar:
        batch_x = batch_x.to(DEVICE, non_blocking=True)

        if DENOISE_STD > 0:
            noisy_x = batch_x + DENOISE_STD * torch.randn_like(batch_x)
        else:
            noisy_x = batch_x

        optimizer.zero_grad()
        recon_x, mu, logvar = model(noisy_x)
        loss, recon_loss, kl_loss = vae_loss_function(recon_x, batch_x, mu, logvar, beta)
        loss.backward()
        optimizer.step()

        epoch_total += loss.item()
        epoch_recon += recon_loss.item()
        epoch_kl += kl_loss.item()

        progress_bar.set_postfix({
            "loss": f"{loss.item()/batch_x.size(0):.4f}",
            "recon": f"{recon_loss.item()/batch_x.size(0):.4f}",
            "kl": f"{kl_loss.item()/batch_x.size(0):.4f}",
            "beta": f"{beta:.4f}"
        })

    avg_total = epoch_total / len(train_loader.dataset)
    avg_recon = epoch_recon / len(train_loader.dataset)
    avg_kl = epoch_kl / len(train_loader.dataset)

    train_losses.append(avg_total)
    train_recons.append(avg_recon)
    train_kls.append(avg_kl)

    # validation clustering every epoch
    Z_val_epoch = extract_latent_mu(model, X_val, DEVICE)
    km_val = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=20)
    val_labels = km_val.fit_predict(Z_val_epoch)
    val_metrics = compute_metrics(Z_val_epoch, val_labels, y_val)
    val_metrics["epoch"] = epoch + 1
    epoch_validation_rows.append(val_metrics)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Total={avg_total:.4f} | Recon={avg_recon:.4f} | KL={avg_kl:.4f} | "
        f"Val ARI={val_metrics['ari']:.4f} | Val NMI={val_metrics['nmi']:.4f}"
    )

torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "final_report_vae_model.pth"))


# ============================================================
# 6. EXTRACT LATENT FEATURES
# ============================================================

Z_all = extract_latent_mu(model, X_scaled, DEVICE)
Z_train = extract_latent_mu(model, X_train, DEVICE)
Z_val = extract_latent_mu(model, X_val, DEVICE)

latent_df = pd.DataFrame(Z_all, columns=[f"z{i+1}" for i in range(Z_all.shape[1])])
latent_df["true_label_encoded"] = y
latent_df["true_label_name"] = y_text
latent_df.to_csv(os.path.join(OUTPUT_DIR, "vae_latent_features.csv"), index=False)


# ============================================================
# 7. CLUSTERING AND BASELINES
# ============================================================

# VAE + KMeans
kmeans_vae = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=20)
vae_kmeans_labels = kmeans_vae.fit_predict(Z_all)
vae_metrics = compute_metrics(Z_all, vae_kmeans_labels, y)

# RAW + GMM baseline
gmm_raw = GaussianMixture(n_components=N_CLUSTERS, random_state=RANDOM_STATE, covariance_type="full")
raw_gmm_labels = gmm_raw.fit_predict(X_scaled)
raw_metrics = compute_metrics(X_scaled, raw_gmm_labels, y)

# PCA + KMeans baseline
pca = PCA(n_components=PCA_LATENT_DIM, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)
kmeans_pca = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=20)
pca_kmeans_labels = kmeans_pca.fit_predict(X_pca)
pca_metrics = compute_metrics(X_pca, pca_kmeans_labels, y)

results_df = pd.DataFrame([
    {
        "Method": "VAE + KMeans",
        "Silhouette": vae_metrics["silhouette"],
        "Calinski_Harabasz": vae_metrics["calinski_harabasz"],
        "ARI": vae_metrics["ari"],
        "NMI": vae_metrics["nmi"]
    },
    {
        "Method": "RAW + GMM",
        "Silhouette": raw_metrics["silhouette"],
        "Calinski_Harabasz": raw_metrics["calinski_harabasz"],
        "ARI": raw_metrics["ari"],
        "NMI": raw_metrics["nmi"]
    },
    {
        "Method": "PCA + KMeans",
        "Silhouette": pca_metrics["silhouette"],
        "Calinski_Harabasz": pca_metrics["calinski_harabasz"],
        "ARI": pca_metrics["ari"],
        "NMI": pca_metrics["nmi"]
    }
])

results_df.to_csv(os.path.join(OUTPUT_DIR, "method_comparison_metrics.csv"), index=False)
print("\nMethod comparison:")
print(results_df)


# ============================================================
# 8. SAVE ASSIGNMENTS AND TABLES
# ============================================================

assignments_df = pd.DataFrame({
    "true_label_encoded": y,
    "true_label_name": y_text,
    "vae_kmeans_cluster": vae_kmeans_labels,
    "raw_gmm_cluster": raw_gmm_labels,
    "pca_kmeans_cluster": pca_kmeans_labels
})
assignments_df.to_csv(os.path.join(OUTPUT_DIR, "cluster_assignments_all_methods.csv"), index=False)

vae_ctab = pd.crosstab(
    pd.Series(vae_kmeans_labels, name="Cluster"),
    pd.Series(y_text, name="True Genre")
)
vae_ctab.to_csv(os.path.join(OUTPUT_DIR, "vae_cluster_vs_genre.csv"))

raw_ctab = pd.crosstab(
    pd.Series(raw_gmm_labels, name="Cluster"),
    pd.Series(y_text, name="True Genre")
)
raw_ctab.to_csv(os.path.join(OUTPUT_DIR, "raw_gmm_cluster_vs_genre.csv"))

pca_ctab = pd.crosstab(
    pd.Series(pca_kmeans_labels, name="Cluster"),
    pd.Series(y_text, name="True Genre")
)
pca_ctab.to_csv(os.path.join(OUTPUT_DIR, "pca_cluster_vs_genre.csv"))


# ============================================================
# 9. REPORT FIGURES
# ============================================================

# Training history
history_df = pd.DataFrame({
    "epoch": np.arange(1, EPOCHS + 1),
    "total_loss": train_losses,
    "recon_loss": train_recons,
    "kl_loss": train_kls,
    "beta": beta_history
})
history_df.to_csv(os.path.join(OUTPUT_DIR, "training_history.csv"), index=False)

val_history_df = pd.DataFrame(epoch_validation_rows)
val_history_df.to_csv(os.path.join(OUTPUT_DIR, "validation_metrics_by_epoch.csv"), index=False)

plot_training_curves(
    train_losses,
    train_recons,
    train_kls,
    beta_history,
    os.path.join(OUTPUT_DIR, "training_curves.png")
)

plot_metric_bars(
    results_df,
    os.path.join(OUTPUT_DIR, "method_comparison_bar_chart.png")
)

plot_cluster_vs_genre_heatmap(
    vae_ctab,
    os.path.join(OUTPUT_DIR, "vae_cluster_vs_genre_heatmap.png"),
    "VAE + KMeans: Cluster vs True Genre"
)

plot_cluster_vs_genre_heatmap(
    raw_ctab,
    os.path.join(OUTPUT_DIR, "raw_gmm_cluster_vs_genre_heatmap.png"),
    "RAW + GMM: Cluster vs True Genre"
)

plot_cluster_vs_genre_heatmap(
    pca_ctab,
    os.path.join(OUTPUT_DIR, "pca_cluster_vs_genre_heatmap.png"),
    "PCA + KMeans: Cluster vs True Genre"
)

# t-SNE
print("\nRunning t-SNE...")
tsne_vae = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=30)
Z_tsne = tsne_vae.fit_transform(Z_all)
pd.DataFrame({
    "tsne_1": Z_tsne[:, 0],
    "tsne_2": Z_tsne[:, 1],
    "true_label_name": y_text,
    "vae_cluster": vae_kmeans_labels
}).to_csv(os.path.join(OUTPUT_DIR, "vae_tsne_coordinates.csv"), index=False)

plot_2d_embedding(
    Z_tsne, y,
    "VAE Latent t-SNE (True Genre)",
    os.path.join(OUTPUT_DIR, "vae_tsne_true_genre.png"),
    class_names=label_encoder.classes_
)

plot_2d_embedding(
    Z_tsne, vae_kmeans_labels,
    "VAE Latent t-SNE (K-Means Clusters)",
    os.path.join(OUTPUT_DIR, "vae_tsne_clusters.png")
)

# PCA t-SNE
tsne_pca = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=30)
X_pca_tsne = tsne_pca.fit_transform(X_pca)
pd.DataFrame({
    "tsne_1": X_pca_tsne[:, 0],
    "tsne_2": X_pca_tsne[:, 1],
    "true_label_name": y_text,
    "pca_cluster": pca_kmeans_labels
}).to_csv(os.path.join(OUTPUT_DIR, "pca_tsne_coordinates.csv"), index=False)

plot_2d_embedding(
    X_pca_tsne, pca_kmeans_labels,
    "PCA t-SNE (K-Means Clusters)",
    os.path.join(OUTPUT_DIR, "pca_tsne_clusters.png")
)

# RAW t-SNE
tsne_raw = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=30)
X_raw_tsne = tsne_raw.fit_transform(X_scaled)
pd.DataFrame({
    "tsne_1": X_raw_tsne[:, 0],
    "tsne_2": X_raw_tsne[:, 1],
    "true_label_name": y_text,
    "raw_cluster": raw_gmm_labels
}).to_csv(os.path.join(OUTPUT_DIR, "raw_tsne_coordinates.csv"), index=False)

plot_2d_embedding(
    X_raw_tsne, raw_gmm_labels,
    "RAW Features t-SNE (GMM Clusters)",
    os.path.join(OUTPUT_DIR, "raw_tsne_clusters.png")
)

# UMAP
if UMAP_AVAILABLE:
    print("Running UMAP...")

    umap_vae = umap.UMAP(n_components=2, random_state=RANDOM_STATE)
    Z_umap = umap_vae.fit_transform(Z_all)
    pd.DataFrame({
        "umap_1": Z_umap[:, 0],
        "umap_2": Z_umap[:, 1],
        "true_label_name": y_text,
        "vae_cluster": vae_kmeans_labels
    }).to_csv(os.path.join(OUTPUT_DIR, "vae_umap_coordinates.csv"), index=False)

    plot_2d_embedding(
        Z_umap, y,
        "VAE Latent UMAP (True Genre)",
        os.path.join(OUTPUT_DIR, "vae_umap_true_genre.png"),
        class_names=label_encoder.classes_
    )

    plot_2d_embedding(
        Z_umap, vae_kmeans_labels,
        "VAE Latent UMAP (K-Means Clusters)",
        os.path.join(OUTPUT_DIR, "vae_umap_clusters.png")
    )

    umap_pca = umap.UMAP(n_components=2, random_state=RANDOM_STATE)
    X_pca_umap = umap_pca.fit_transform(X_pca)
    plot_2d_embedding(
        X_pca_umap, pca_kmeans_labels,
        "PCA UMAP (K-Means Clusters)",
        os.path.join(OUTPUT_DIR, "pca_umap_clusters.png")
    )

    umap_raw = umap.UMAP(n_components=2, random_state=RANDOM_STATE)
    X_raw_umap = umap_raw.fit_transform(X_scaled)
    plot_2d_embedding(
        X_raw_umap, raw_gmm_labels,
        "RAW Features UMAP (GMM Clusters)",
        os.path.join(OUTPUT_DIR, "raw_umap_clusters.png")
    )


# ============================================================
# 10. RUN SUMMARY
# ============================================================

summary = {
    "feature_file_used": feature_file,
    "device_used": str(DEVICE),
    "cuda_available": bool(torch.cuda.is_available()),
    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    "final_model_config": {
        "latent_dim": LATENT_DIM,
        "hidden_dims": HIDDEN_DIMS,
        "beta_max": BETA_MAX,
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "warmup_epochs": WARMUP_EPOCHS,
        "use_batchnorm": USE_BATCHNORM,
        "denoise_std": DENOISE_STD
    },
    "final_metrics": {
        "vae_kmeans": vae_metrics,
        "raw_gmm": raw_metrics,
        "pca_kmeans": pca_metrics
    }
}
save_json(summary, os.path.join(OUTPUT_DIR, "run_summary.json"))

print("\nAll report-ready outputs saved to:")
print(OUTPUT_DIR)