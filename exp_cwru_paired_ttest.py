import os
import glob
import re
import warnings
import scipy.io as sio
import numpy as np
import scipy.stats as stats
import matplotlib
import matplotlib.pyplot as plt
import pywt
from scipy import signal
from scipy.ndimage import zoom
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

matplotlib.use('TkAgg')
warnings.filterwarnings('ignore')

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
                        X_tr.append(window_sig); y_tr.append(lbl)
                    elif partition == 'test':
                        X_te.append(window_sig); y_te.append(lbl)
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
    def __init__(self, hidden_dims, num_classes=4):
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
    def __init__(self, ts, feats, labels=None):
        self.ts = torch.FloatTensor(ts)
        self.labels = torch.LongTensor(labels) if labels is not None else torch.zeros(len(ts), dtype=torch.long)
        self.stft, self.wt, self.gaf = (torch.FloatTensor(feats[k]) for k in ['stft', 'wavelet', 'gaf'])

    def __len__(self): return len(self.ts)

    def __getitem__(self, idx): return {'ts': self.ts[idx], 'stft': self.stft[idx], 'wt': self.wt[idx],
                                        'gaf': self.gaf[idx], 'lbl': self.labels[idx]}


def evaluate(model, loader):
    model.eval()
    preds, labels_all, probs_all = [], [], []
    with torch.no_grad():
        for b in loader:
            ts, stft, wt, gaf, lbl = (b[k].to(device) for k in ['ts', 'stft', 'wt', 'gaf', 'lbl'])
            out = model(ts, stft, wt, gaf)
            probs_all.extend(F.softmax(out, dim=1).cpu().numpy())
            preds.extend(torch.argmax(out, dim=1).cpu().numpy())
            labels_all.extend(lbl.cpu().numpy())
    acc = accuracy_score(labels_all, preds)
    try:
        auc_val = roc_auc_score(labels_all, probs_all, multi_class='ovr', average='macro')
    except Exception:
        auc_val = 0.5
    return acc, auc_val


def run_few_shot_experiment():
    X, y, idx_tr_full, idx_te = load_cwru_data_strict(DATA_DIR, seq_len=1024)
    feats = extract_features(X)
    N_SHOTS_PER_CLASS = 15
    SEEDS = [42, 1024, 2024, 2026, 3407]
    acc_baseline, auc_baseline, acc_proposed, auc_proposed = [], [], [], []

    for i, seed in enumerate(SEEDS):
        torch.manual_seed(seed)
        np.random.seed(seed)

        idx_tr_labeled = idx_tr_full[next(
            StratifiedShuffleSplit(n_splits=1, train_size=N_SHOTS_PER_CLASS * 4, random_state=seed).split(
                idx_tr_full, y[idx_tr_full]))[0]]
        loader_unlabeled = DataLoader(
            MultiModalDataset(X[idx_tr_full], {k: feats[k][idx_tr_full] for k in feats}, y[idx_tr_full]), batch_size=64,
            shuffle=True)
        loader_labeled = DataLoader(
            MultiModalDataset(X[idx_tr_labeled], {k: feats[k][idx_tr_labeled] for k in feats}, y[idx_tr_labeled]),
            batch_size=16, shuffle=True)
        loader_test = DataLoader(MultiModalDataset(X[idx_te], {k: feats[k][idx_te] for k in feats}, y[idx_te]),
                                 batch_size=64, shuffle=False)
        ce_loss = nn.CrossEntropyLoss()

        # Baseline Model (Lightweight architecture trained from scratch)
        base_model = MultiModalNet([256, 128]).to(device)
        opt_base = torch.optim.Adam(base_model.parameters(), lr=1e-3)
        for ep in range(15):
            base_model.train()
            for b in loader_labeled:
                ts, stft, wt, gaf, lbl = (b[k].to(device) for k in ['ts', 'stft', 'wt', 'gaf', 'lbl'])
                opt_base.zero_grad()
                ce_loss(base_model(ts, stft, wt, gaf), lbl).backward()
                opt_base.step()
        b_acc, b_auc = evaluate(base_model, loader_test)
        acc_baseline.append(b_acc)
        auc_baseline.append(b_auc)

        # Proposed Framework (SSL Pre-training + Fine-tuning + KD)
        teacher = MultiModalNet([512, 256, 128]).to(device)
        opt_pre = torch.optim.Adam(teacher.parameters(), lr=1e-3)
        ssl_loss_fn = ContrastiveLoss(temperature=0.1)
        for ep in range(5):
            teacher.train()
            for b in loader_unlabeled:
                ts, stft, wt, gaf = (b[k].to(device) for k in ['ts', 'stft', 'wt', 'gaf'])
                ts1, stft1, wt1, gaf1 = augment_batch(ts, stft, wt, gaf)
                ts2, stft2, wt2, gaf2 = augment_batch(ts, stft, wt, gaf)
                loss = ssl_loss_fn(
                    torch.cat([teacher(ts1, stft1, wt1, gaf1, True), teacher(ts2, stft2, wt2, gaf2, True)], dim=0))
                opt_pre.zero_grad()
                loss.backward()
                opt_pre.step()

        opt_ft = torch.optim.Adam(teacher.parameters(), lr=1e-3)
        for ep in range(15):
            teacher.train()
            for b in loader_labeled:
                ts, stft, wt, gaf, lbl = (b[k].to(device) for k in ['ts', 'stft', 'wt', 'gaf', 'lbl'])
                opt_ft.zero_grad()
                ce_loss(teacher(ts, stft, wt, gaf), lbl).backward()
                opt_ft.step()

        student = MultiModalNet([256, 128]).to(device)
        opt_kd = torch.optim.Adam(student.parameters(), lr=1e-3)
        kl_loss = nn.KLDivLoss(reduction='batchmean')
        T, alpha = 3.0, 0.6
        for ep in range(15):
            student.train()
            for b in loader_labeled:
                ts, stft, wt, gaf, lbl = (b[k].to(device) for k in ['ts', 'stft', 'wt', 'gaf', 'lbl'])
                opt_kd.zero_grad()
                s_log = student(ts, stft, wt, gaf)
                with torch.no_grad(): t_log = teacher(ts, stft, wt, gaf)
                loss = alpha * (kl_loss(F.log_softmax(s_log / T, dim=1), F.softmax(t_log / T, dim=1)) * (T ** 2)) + (
                            1 - alpha) * ce_loss(s_log, lbl)
                loss.backward()
                opt_kd.step()

        p_acc, p_auc = evaluate(student, loader_test)
        acc_proposed.append(p_acc)
        auc_proposed.append(p_auc)
        print(f"Run {i + 1} => Base: {b_acc:.4f} | Prop: {p_acc:.4f}")

    mean_b_acc, std_b_acc = np.mean(acc_baseline), np.std(acc_baseline)
    mean_p_acc, std_p_acc = np.mean(acc_proposed), np.std(acc_proposed)
    t_acc, p_acc_val = stats.ttest_rel(acc_proposed, acc_baseline)
    print("\n=== FINAL PAPER REPORT ===")
    print(f"Baseline: {mean_b_acc * 100:.2f}% ± {std_b_acc * 100:.2f}%")
    print(f"Proposed: {mean_p_acc * 100:.2f}% ± {std_p_acc * 100:.2f}%")
    print(f"Paired T-test P-value: {p_acc_val / 2 if t_acc > 0 else 1.0:.4e}")


if __name__ == "__main__":
    run_few_shot_experiment()