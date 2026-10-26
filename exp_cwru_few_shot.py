import os
import glob
import re
import warnings
import scipy.io as sio
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pywt
from scipy import signal
from scipy.ndimage import zoom
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

matplotlib.use('TkAgg')
warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

DATA_DIR = r"./data/cwru_data"
OUTPUT_DIR = r"./results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def load_cwru_data_strict(data_dir, seq_len=1024):
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dir not found: {data_dir}")
    mat_files = glob.glob(os.path.join(data_dir, '**', '*.mat'), recursive=True)

    train_list = [97, 98, 105, 106, 118, 119, 130, 131, 144, 145, 156, 158]
    val_list = [99, 107, 120, 132, 146, 159]
    test_list = [100, 108, 121, 133, 147, 160]

    def get_label(num):
        if num in [97, 98, 99, 100]: return 0
        if num in [105, 106, 107, 108, 169, 170, 171, 172]: return 1
        if num in [130, 131, 132, 133, 144, 145, 146, 147, 156, 158, 159, 160]: return 2
        if num in [118, 119, 120, 121, 185, 186, 187, 188]: return 3
        return -1

    X_tr, y_tr, X_te, y_te = [], [], [], []

    for file_path in mat_files:
        filename = os.path.basename(file_path)
        match = re.search(r'(\d+)', filename)
        if not match: continue
        num = int(match.group(1))

        lbl = get_label(num)
        if lbl == -1: continue

        partition = None
        if num in train_list or num in val_list:
            partition = 'train'
        elif num in test_list:
            partition = 'test'

        if partition is None: continue

        try:
            mat_dict = sio.loadmat(file_path)
            sig = None
            for key in mat_dict.keys():
                if 'DE_time' in key:
                    sig = mat_dict[key].flatten()
                    break
            if sig is None:
                for key in mat_dict.keys():
                    if not key.startswith('__') and isinstance(mat_dict[key], np.ndarray):
                        sig = mat_dict[key].flatten()
                        break
            if sig is not None:
                num_windows = len(sig) // seq_len
                for i in range(num_windows):
                    window_sig = sig[i * seq_len: (i + 1) * seq_len]
                    window_sig = (window_sig - np.mean(window_sig)) / (np.std(window_sig) + 1e-8)
                    if partition == 'train':
                        X_tr.append(window_sig)
                        y_tr.append(lbl)
                    elif partition == 'test':
                        X_te.append(window_sig)
                        y_te.append(lbl)
        except Exception:
            pass

    X_tr, y_tr = np.array(X_tr), np.array(y_tr)
    X_te, y_te = np.array(X_te), np.array(y_te)
    X = np.concatenate([X_tr, X_te], axis=0)
    idx_tr = np.arange(len(X_tr))
    idx_te = np.arange(len(X_tr), len(X))

    return X, np.concatenate([y_tr, y_te], axis=0), idx_tr, idx_te


def extract_features(signals, img_size=(32, 32)):
    features = {'stft': [], 'wavelet': [], 'gaf': []}
    scales = np.arange(1, 65)

    def _resize(img):
        img = img.reshape(-1, 1) if img.ndim == 1 else img
        resized = zoom(img, (img_size[0] / img.shape[0], img_size[1] / img.shape[1]))
        h, w = resized.shape
        return np.pad(resized[:img_size[0], :img_size[1]], ((0, max(0, img_size[0] - h)), (0, max(0, img_size[1] - w))),
                      mode='constant')

    for sig in signals:
        _, _, Zxx = signal.stft(sig, fs=12000, nperseg=128)
        features['stft'].append(_resize(np.abs(Zxx)))
        coeffs, _ = pywt.cwt(sig, scales, 'morl')
        features['wavelet'].append(_resize(np.abs(coeffs)))
        s_norm = (sig - np.min(sig)) / (np.max(sig) - np.min(sig) + 1e-8)
        phi = np.arccos(np.clip(s_norm, -1, 1))
        features['gaf'].append(_resize(np.outer(np.cos(phi), np.cos(phi)) - np.outer(np.sin(phi), np.sin(phi))))
    return {k: np.array(v) for k, v in features.items()}


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=2)

    def forward(self, projections):
        batch_size = projections.size(0) // 2
        sim_matrix = self.cosine_similarity(projections.unsqueeze(1), projections.unsqueeze(0)) / self.temperature
        loss = 0
        for i in range(2 * batch_size):
            pos_idx = (i + batch_size) % (2 * batch_size) if i < batch_size else i - batch_size
            pos_sim = sim_matrix[i, pos_idx]
            mask = torch.ones(2 * batch_size, dtype=torch.bool, device=projections.device)
            mask[i] = False
            mask[pos_idx] = False
            neg_sims = sim_matrix[i, mask]
            numerator = torch.exp(pos_sim)
            denominator = numerator + torch.sum(torch.exp(neg_sims))
            loss += -torch.log(numerator / denominator)
        return loss / (2 * batch_size)


def augment_batch(ts, stft, wt, gaf):
    noise_factor = 0.02
    scale_factor = torch.rand(ts.size(0), 1, device=ts.device) * 0.2 + 0.9
    ts_aug = ts * scale_factor + torch.randn_like(ts) * noise_factor
    stft_aug = stft + torch.randn_like(stft) * noise_factor
    wt_aug = wt + torch.randn_like(wt) * noise_factor
    gaf_aug = gaf + torch.randn_like(gaf) * noise_factor
    return ts_aug, stft_aug, wt_aug, gaf_aug


class ConvEncoder(nn.Module):
    def __init__(self, is_1d=False, in_c=1, rep_dim=128):
        super().__init__()
        self.is_1d = is_1d
        if is_1d:
            self.net = nn.Sequential(nn.Conv1d(in_c, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
                                     nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(16), nn.Flatten(),
                                     nn.Linear(64 * 16, rep_dim))
        else:
            self.net = nn.Sequential(nn.Conv2d(in_c, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                     nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)),
                                     nn.Flatten(), nn.Linear(64 * 16, rep_dim))

    def forward(self, x):
        if self.is_1d and x.dim() == 2: x = x.unsqueeze(1)
        if not self.is_1d:
            x = x.unsqueeze(0).unsqueeze(0) if x.dim() == 2 else x.unsqueeze(1)
            if x.size(2) < 4: x = F.interpolate(x, size=(8, 8), mode='bilinear')
        return self.net(x)


class MultiModalNet(nn.Module):
    def __init__(self, hidden_dims=[512, 256, 128], num_classes=4):
        super().__init__()
        self.encoders = nn.ModuleList([ConvEncoder(is_1d=True)] + [ConvEncoder() for _ in range(3)])
        self.projector = nn.Sequential(nn.Linear(128 * 4, 256), nn.ReLU(), nn.Linear(256, 128))
        layers = []
        in_dim = 128 * 4
        for h in hidden_dims: layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(0.3)]); in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, ts, stft, wt, gaf, return_proj=False):
        fused = torch.cat([enc(x) for enc, x in zip(self.encoders, [ts, stft, wt, gaf])], dim=1)
        if return_proj: return self.projector(fused)
        return self.classifier(fused)


class MultiModalDataset(Dataset):
    def __init__(self, ts, feats, labels):
        self.ts, self.labels = torch.FloatTensor(ts), torch.LongTensor(labels)
        self.stft, self.wt, self.gaf = (torch.FloatTensor(feats[k]) for k in ['stft', 'wavelet', 'gaf'])

    def __len__(self): return len(self.ts)

    def __getitem__(self, idx): return {'ts': self.ts[idx], 'stft': self.stft[idx], 'wt': self.wt[idx],
                                        'gaf': self.gaf[idx], 'lbl': self.labels[idx]}


def evaluate_model(model, loader):
    model.eval()
    preds, labels_all = [], []
    with torch.no_grad():
        for b in loader:
            ts, stft, wt, gaf, lbl = (b[k].to(device) for k in ['ts', 'stft', 'wt', 'gaf', 'lbl'])
            preds.extend(torch.argmax(model(ts, stft, wt, gaf), dim=1).cpu().numpy())
            labels_all.extend(lbl.cpu().numpy())
    return accuracy_score(labels_all, preds)


def plot_few_label_results(fractions, acc_baseline, acc_ssl, save_path):
    plt.figure(figsize=(7, 5))
    x_labels = [f"{int(frac * 100)}%" for frac in fractions]
    plt.plot(x_labels, acc_baseline, marker='o', linestyle='--', color='#d62728', linewidth=2, label='Baseline')
    plt.plot(x_labels, acc_ssl, marker='s', linestyle='-', color='#1f77b4', linewidth=2, label='Proposed')
    plt.title('Few-Shot Evaluation (5-Seed Avg)', fontsize=12, pad=12)
    plt.xlabel('Labeled Data Fraction', fontsize=11)
    plt.ylabel('Accuracy', fontsize=11)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='lower right', fontsize=10)
    for i, (b, s) in enumerate(zip(acc_baseline, acc_ssl)):
        plt.annotate(f'{b:.3f}', (i, b), textcoords="offset points", xytext=(0, -12), ha='center', fontsize=8,
                     color='#d62728')
        plt.annotate(f'{s:.3f}', (i, s), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=8,
                     color='#1f77b4')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    print("=== Load Data ===")
    X, y, idx_tr, idx_te = load_cwru_data_strict(DATA_DIR, seq_len=1024)
    print("=== Extract Features ===")
    feats = extract_features(X)

    dataset_te = MultiModalDataset(X[idx_te], {k: feats[k][idx_te] for k in feats}, y[idx_te])
    loader_te = DataLoader(dataset_te, batch_size=64, shuffle=False)
    dataset_tr_unlabeled = MultiModalDataset(X[idx_tr], {k: feats[k][idx_tr] for k in feats}, y[idx_tr])
    loader_tr_unlabeled = DataLoader(dataset_tr_unlabeled, batch_size=64, shuffle=True)

    label_fractions = [0.01, 0.05, 0.10, 0.20, 1.0]
    SEEDS = [42, 137, 256, 512, 1024]

    acc_results_baseline = []
    acc_results_ssl = []

    print("\n=== Few-Label Evaluation (5-Seed) ===")
    for frac in label_fractions:
        b_acc_seeds, s_acc_seeds = [], []

        for seed in SEEDS:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

            if frac == 1.0:
                idx_labeled = idx_tr
            else:
                sss = StratifiedShuffleSplit(n_splits=1, train_size=frac, random_state=seed)
                for idx_split, _ in sss.split(idx_tr, y[idx_tr]):
                    idx_labeled = idx_tr[idx_split]

            dataset_tr_labeled = MultiModalDataset(X[idx_labeled], {k: feats[k][idx_labeled] for k in feats},
                                                   y[idx_labeled])
            loader_tr_labeled = DataLoader(dataset_tr_labeled, batch_size=32, shuffle=True)

            # Baseline (Supervised only)
            model_base = MultiModalNet().to(device)
            opt_base = torch.optim.Adam(model_base.parameters(), lr=1e-3)
            ce_loss = nn.CrossEntropyLoss()
            epochs_ft = 10 if frac < 0.2 else 5
            for ep in range(epochs_ft):
                model_base.train()
                for b in loader_tr_labeled:
                    ts, stft, wt, gaf, lbl = (b[k].to(device) for k in ['ts', 'stft', 'wt', 'gaf', 'lbl'])
                    opt_base.zero_grad()
                    ce_loss(model_base(ts, stft, wt, gaf), lbl).backward()
                    opt_base.step()
            b_acc_seeds.append(evaluate_model(model_base, loader_te))

            # Proposed (Self-supervised pre-training + Fine-tuning)
            model_ssl = MultiModalNet().to(device)
            opt_pre = torch.optim.Adam(model_ssl.parameters(), lr=1e-3)
            contrastive_criterion = ContrastiveLoss(0.1)
            for ep in range(10):
                model_ssl.train()
                for b in loader_tr_unlabeled:
                    ts, stft, wt, gaf = (b[k].to(device) for k in ['ts', 'stft', 'wt', 'gaf'])
                    ts1, stft1, wt1, gaf1 = augment_batch(ts, stft, wt, gaf)
                    ts2, stft2, wt2, gaf2 = augment_batch(ts, stft, wt, gaf)
                    loss = contrastive_criterion(torch.cat([
                        model_ssl(ts1, stft1, wt1, gaf1, True),
                        model_ssl(ts2, stft2, wt2, gaf2, True)
                    ], dim=0))
                    opt_pre.zero_grad()
                    loss.backward()
                    opt_pre.step()

            opt_ft = torch.optim.Adam(model_ssl.parameters(), lr=5e-4)
            for ep in range(epochs_ft):
                model_ssl.train()
                for b in loader_tr_labeled:
                    ts, stft, wt, gaf, lbl = (b[k].to(device) for k in ['ts', 'stft', 'wt', 'gaf', 'lbl'])
                    opt_ft.zero_grad()
                    ce_loss(model_ssl(ts, stft, wt, gaf), lbl).backward()
                    opt_ft.step()
            s_acc_seeds.append(evaluate_model(model_ssl, loader_te))

        mean_b, std_b = np.mean(b_acc_seeds), np.std(b_acc_seeds)
        mean_s, std_s = np.mean(s_acc_seeds), np.std(s_acc_seeds)

        acc_results_baseline.append(mean_b)
        acc_results_ssl.append(mean_s)

        print(f"\n[ Fraction: {frac * 100}% ]")
        print(f"  Runs Base: {[round(x, 3) for x in b_acc_seeds]}")
        print(f"  Runs Prop: {[round(x, 3) for x in s_acc_seeds]}")
        print(f"  => Stats Base: {mean_b:.3f} ± {std_b:.3f} | Prop: {mean_s:.3f} ± {std_s:.3f}")

    plot_path = os.path.join(OUTPUT_DIR, "Few_Label_Robustness_CWRU_EN.png")
    plot_few_label_results(label_fractions, acc_results_baseline, acc_results_ssl, plot_path)
    print(f"\n=== Done! Chart saved: {plot_path} ===")


if __name__ == "__main__":
    main()