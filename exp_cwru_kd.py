import os
import glob
import re
import warnings
import scipy.io as sio
import numpy as np
import pywt
from scipy import signal
from scipy.ndimage import zoom
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')

# Fixed random seed for controlled ablation (Seed: 2026)
SEED = 2026
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

DATA_DIR = r"./data/cwru_data"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def load_cwru_data_strict(data_dir, seq_len=1024):
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

        partition = 'train' if num in train_list or num in val_list else ('test' if num in test_list else None)
        if not partition: continue

        try:
            mat_dict = sio.loadmat(file_path)
            sig = next((mat_dict[k].flatten() for k in mat_dict.keys() if 'DE_time' in k), None)
            if sig is None:
                sig = next((mat_dict[k].flatten() for k in mat_dict.keys() if
                            not k.startswith('__') and isinstance(mat_dict[k], np.ndarray)), None)
            if sig is not None:
                for i in range(len(sig) // seq_len):
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

    X = np.concatenate([np.array(X_tr), np.array(X_te)], axis=0)
    y = np.concatenate([np.array(y_tr), np.array(y_te)], axis=0)
    idx_tr = np.arange(len(X_tr))
    idx_te = np.arange(len(X_tr), len(X))
    return X, y, idx_tr, idx_te


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
        super().__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=2)

    def forward(self, projections):
        batch_size = projections.size(0) // 2
        sim_matrix = self.cosine_similarity(projections.unsqueeze(1), projections.unsqueeze(0)) / self.temperature
        loss = 0
        for i in range(2 * batch_size):
            pos_idx = (i + batch_size) % (2 * batch_size) if i < batch_size else i - batch_size
            mask = torch.ones(2 * batch_size, dtype=torch.bool, device=projections.device)
            mask[i], mask[pos_idx] = False, False
            loss += -torch.log(torch.exp(sim_matrix[i, pos_idx]) / (
                        torch.exp(sim_matrix[i, pos_idx]) + torch.sum(torch.exp(sim_matrix[i, mask]))))
        return loss / (2 * batch_size)


def augment_batch(ts, stft, wt, gaf):
    noise_factor = 0.05
    scale_factor = torch.rand(ts.size(0), 1, device=ts.device) * 0.4 + 0.8
    return ts * scale_factor + torch.randn_like(ts) * noise_factor, stft + torch.randn_like(
        stft) * noise_factor, wt + torch.randn_like(wt) * noise_factor, gaf + torch.randn_like(gaf) * noise_factor


class ConvEncoder(nn.Module):
    def __init__(self, is_1d=False, in_c=1, rep_dim=128):
        super().__init__()
        self.is_1d = is_1d
        if is_1d:
            self.net = nn.Sequential(
                nn.Conv1d(in_c, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.AdaptiveAvgPool1d(16),
                nn.Flatten(), nn.Linear(64 * 16, rep_dim)
            )
        else:
            self.net = nn.Sequential(
                nn.Conv2d(in_c, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(), nn.Linear(64 * 16, rep_dim)
            )

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


def evaluate_metrics(model, loader):
    model.eval()
    preds, labels_all = [], []
    with torch.no_grad():
        for b in loader:
            ts, stft, wt, gaf, lbl = (b[k].to(device) for k in ['ts', 'stft', 'wt', 'gaf', 'lbl'])
            preds.extend(torch.argmax(model(ts, stft, wt, gaf), dim=1).cpu().numpy())
            labels_all.extend(lbl.cpu().numpy())
    acc = accuracy_score(labels_all, preds)
    f1 = f1_score(labels_all, preds, average='weighted', zero_division=0)
    return acc, f1


def main():
    print("=== Loading CWRU Dataset for Table 1 Ablation ===")
    X, y, idx_tr_full, idx_te = load_cwru_data_strict(DATA_DIR, seq_len=1024)
    feats = extract_features(X)

    # Sample 5% labeled subset from training data (Seed: 2026)
    sss = StratifiedShuffleSplit(n_splits=1, train_size=int(len(idx_tr_full) * 0.05), random_state=SEED)
    for idx_subset, _ in sss.split(idx_tr_full, y[idx_tr_full]):
        idx_5pct = idx_tr_full[idx_subset]

    loader_unlabeled = DataLoader(
        MultiModalDataset(X[idx_tr_full], {k: feats[k][idx_tr_full] for k in feats}, y[idx_tr_full]), batch_size=64,
        shuffle=True)
    loader_labeled_5pct = DataLoader(
        MultiModalDataset(X[idx_5pct], {k: feats[k][idx_5pct] for k in feats}, y[idx_5pct]), batch_size=16,
        shuffle=True)
    loader_te = DataLoader(MultiModalDataset(X[idx_te], {k: feats[k][idx_te] for k in feats}, y[idx_te]), batch_size=64,
                           shuffle=False)

    print("\n--- Training Teacher Model (Hidden: [512, 256, 128], 20 Epochs) ---")
    teacher = MultiModalNet([512, 256, 128]).to(device)

    # 1. Self-supervised pre-training (5 Epochs)
    opt_pre = torch.optim.Adam(teacher.parameters(), lr=1e-3)
    contrastive_criterion = ContrastiveLoss(0.1)
    for ep in range(5):
        teacher.train()
        for b in loader_unlabeled:
            ts, stft, wt, gaf = (b[k].to(device) for k in ['ts', 'stft', 'wt', 'gaf'])
            ts1, stft1, wt1, gaf1 = augment_batch(ts, stft, wt, gaf)
            ts2, stft2, wt2, gaf2 = augment_batch(ts, stft, wt, gaf)
            loss = contrastive_criterion(
                torch.cat([teacher(ts1, stft1, wt1, gaf1, True), teacher(ts2, stft2, wt2, gaf2, True)], dim=0))
            opt_pre.zero_grad()
            loss.backward()
            opt_pre.step()

    # 2. Full supervised fine-tuning (20 Epochs)
    opt_ft = torch.optim.Adam(teacher.parameters(), lr=1e-3)
    ce_loss = nn.CrossEntropyLoss()
    for ep in range(20):
        teacher.train()
        for b in loader_unlabeled:
            ts, stft, wt, gaf, lbl = (b[k].to(device) for k in ['ts', 'stft', 'wt', 'gaf', 'lbl'])
            opt_ft.zero_grad()
            ce_loss(teacher(ts, stft, wt, gaf), lbl).backward()
            opt_ft.step()
    teacher.eval()
    print("Teacher Training Completed.")

    # Ablation and hyperparameter configurations: (setting_name, Td, alpha)
    ablation_settings = [
        ("Hard labels Student", 3.0, 0.0),
        ("Soft labels only", 3.0, 1.0),
        ("KD (T=1.0, alpha=0.6)", 1.0, 0.6),
        ("KD (T=3.0, alpha=0.6)", 3.0, 0.6),
        ("KD (T=5.0, alpha=0.6)", 5.0, 0.6),
        ("KD (T=10.0, alpha=0.6)", 10.0, 0.6),
        ("KD (T=3.0, alpha=0.2)", 3.0, 0.2),
        ("KD (T=3.0, alpha=0.8)", 3.0, 0.8),
    ]

    print("\n" + "=" * 85)
    print(f"{'Student Training Setting':<30} | {'Td':<6} | {'alpha':<6} | {'Accuracy':<10} | {'Weighted F1':<10}")
    print("=" * 85)

    for name, Td, alpha in ablation_settings:
        # Ultra-lightweight student architecture [64, 32]
        student = MultiModalNet([64, 32]).to(device)
        opt_s = torch.optim.Adam(student.parameters(), lr=1e-3)
        kl_loss = nn.KLDivLoss(reduction='batchmean')

        for ep in range(15):
            student.train()
            for b in loader_labeled_5pct:
                ts, stft, wt, gaf, lbl = (b[k].to(device) for k in ['ts', 'stft', 'wt', 'gaf', 'lbl'])
                opt_s.zero_grad()
                s_log = student(ts, stft, wt, gaf)

                if alpha == 0.0:
                    loss = ce_loss(s_log, lbl)
                elif alpha == 1.0:
                    with torch.no_grad():
                        t_log = teacher(ts, stft, wt, gaf)
                    loss = kl_loss(F.log_softmax(s_log / Td, dim=1), F.softmax(t_log / Td, dim=1)) * (Td ** 2)
                else:
                    with torch.no_grad():
                        t_log = teacher(ts, stft, wt, gaf)
                    loss = alpha * (
                                kl_loss(F.log_softmax(s_log / Td, dim=1), F.softmax(t_log / Td, dim=1)) * (Td ** 2)) + (
                                       1 - alpha) * ce_loss(s_log, lbl)

                loss.backward()
                opt_s.step()

        acc, f1 = evaluate_metrics(student, loader_te)
        print(f"{name:<30} | {str(Td):<6} | {str(alpha):<6} | {acc:.4f}     | {f1:.4f}")

    print("=" * 85)
    print("=== Table 1 Replication Completed! ===")


if __name__ == "__main__":
    main()