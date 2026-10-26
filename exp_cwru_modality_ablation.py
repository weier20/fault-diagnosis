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
from sklearn.metrics import accuracy_score, f1_score

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
    def __init__(self, hidden_dims, active_mods=[True, True, True, True], num_classes=4):
        super().__init__()
        self.encoders = nn.ModuleList([ConvEncoder(is_1d=True)] + [ConvEncoder() for _ in range(3)])
        self.active_mods = active_mods
        in_dim = 128 * sum(active_mods)
        self.projector = nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(), nn.Linear(256, 128))
        layers = []
        for h in hidden_dims: layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(0.3)]); in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, ts, stft, wt, gaf, return_proj=False):
        fused = torch.cat(
            [enc(x) for i, (enc, x) in enumerate(zip(self.encoders, [ts, stft, wt, gaf])) if self.active_mods[i]],
            dim=1)
        if return_proj: return self.projector(fused)
        return self.classifier(fused)


class MultiModalDataset(Dataset):
    def __init__(self, ts, feats, labels):
        self.ts, self.labels = torch.FloatTensor(ts), torch.LongTensor(labels)
        self.stft, self.wt, self.gaf = (torch.FloatTensor(feats[k]) for k in ['stft', 'wavelet', 'gaf'])

    def __len__(self): return len(self.ts)

    def __getitem__(self, idx): return {'ts': self.ts[idx], 'stft': self.stft[idx], 'wt': self.wt[idx],
                                        'gaf': self.gaf[idx], 'lbl': self.labels[idx]}


def plot_ablation_bar(results, save_path):
    names = list(results.keys())
    accs = [results[n]['acc'] for n in names]
    f1s = [results[n]['f1'] for n in names]
    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.bar(x - width / 2, accs, width, label='Accuracy', color='#4c72b0')
    b2 = ax.bar(x + width / 2, f1s, width, label='F1 Score', color='#55a868')
    ax.set_title('CWRU Data: Complete Ablation Study', fontsize=15, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha='right', fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.legend(loc='upper right', fontsize=11)
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3),
                        textcoords="offset points", ha='center', va='bottom', fontsize=9, rotation=90)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_ablation_radar(results, save_path):
    labels = list(results.keys())
    num_vars = len(labels)
    accs = [results[n]['acc'] for n in labels]
    f1s = [results[n]['f1'] for n in labels]
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    accs += accs[:1]
    f1s += f1s[:1]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([l.split('. ', 1)[-1] for l in labels], fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.plot(angles, accs, linewidth=2, label='Accuracy', color='#4c72b0')
    ax.fill(angles, accs, alpha=0.2, color='#4c72b0')
    ax.plot(angles, f1s, linewidth=2, label='F1 Score', color='#55a868')
    ax.fill(angles, f1s, alpha=0.2, color='#55a868')
    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1))
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    X, y, idx_tr, idx_te = load_cwru_data_strict(DATA_DIR, seq_len=1024)
    feats = extract_features(X)

    loader_tr = DataLoader(MultiModalDataset(X[idx_tr], {k: feats[k][idx_tr] for k in feats}, y[idx_tr]), batch_size=32,
                           shuffle=True)
    loader_te = DataLoader(MultiModalDataset(X[idx_te], {k: feats[k][idx_te] for k in feats}, y[idx_te]), batch_size=32,
                           shuffle=False)

    configs = [
        {"name": "1. Full System", "mods": [True, True, True, True], "ssl": True, "kd": True},
        {"name": "2. w/o KD", "mods": [True, True, True, True], "ssl": True, "kd": False},
        {"name": "3. w/o SSL", "mods": [True, True, True, True], "ssl": False, "kd": True},
        {"name": "4. Only 1D TS", "mods": [True, False, False, False], "ssl": True, "kd": True},
        {"name": "5. Only STFT", "mods": [False, True, False, False], "ssl": True, "kd": True},
        {"name": "6. Only WT", "mods": [False, False, True, False], "ssl": True, "kd": True},
        {"name": "7. Only GAF", "mods": [False, False, False, True], "ssl": True, "kd": True},
        {"name": "8. w/o 1D TS", "mods": [False, True, True, True], "ssl": True, "kd": True},
        {"name": "9. w/o STFT", "mods": [True, False, True, True], "ssl": True, "kd": True},
        {"name": "10. w/o WT", "mods": [True, True, False, True], "ssl": True, "kd": True},
        {"name": "11. w/o GAF", "mods": [True, True, True, False], "ssl": True, "kd": True}
    ]

    results = {}
    for cfg in configs:
        torch.manual_seed(42)
        np.random.seed(42)
        print(f"\n>>> Running Ablation: {cfg['name']} <<<")
        teacher = MultiModalNet([512, 256, 128], cfg['mods']).to(device)
        student = MultiModalNet([256, 128], cfg['mods']).to(device)

        if cfg['ssl']:
            opt_pre = torch.optim.Adam(teacher.parameters(), lr=1e-3)
            crit = ContrastiveLoss(0.1)
            for _ in range(5):
                teacher.train()
                for b in loader_tr:
                    ts, stft, wt, gaf = (b[k].to(device) for k in ['ts', 'stft', 'wt', 'gaf'])
                    loss = crit(torch.cat([teacher(*augment_batch(ts, stft, wt, gaf), True),
                                           teacher(*augment_batch(ts, stft, wt, gaf), True)], dim=0))
                    opt_pre.zero_grad()
                    loss.backward()
                    opt_pre.step()

        if cfg['kd']:
            opt_t = torch.optim.Adam(teacher.parameters(), lr=1e-3)
            ce = nn.CrossEntropyLoss()
            for _ in range(5):
                teacher.train()
                for b in loader_tr:
                    ts, stft, wt, gaf, lbl = (b[k].to(device) for k in ['ts', 'stft', 'wt', 'gaf', 'lbl'])
                    opt_t.zero_grad()
                    ce(teacher(ts, stft, wt, gaf), lbl).backward()
                    opt_t.step()

        opt_s = torch.optim.Adam(student.parameters(), lr=1e-3)
        ce = nn.CrossEntropyLoss()
        kl = nn.KLDivLoss(reduction='batchmean')
        T, alpha = 3.0, 0.6

        for _ in range(10):
            student.train()
            for b in loader_tr:
                ts, stft, wt, gaf, lbl = (b[k].to(device) for k in ['ts', 'stft', 'wt', 'gaf', 'lbl'])
                opt_s.zero_grad()
                s_log = student(ts, stft, wt, gaf)
                if cfg['kd']:
                    with torch.no_grad():
                        t_log = teacher(ts, stft, wt, gaf)
                    loss = alpha * (kl(F.log_softmax(s_log / T, dim=1), F.softmax(t_log / T, dim=1)) * (T ** 2)) + (
                                1 - alpha) * ce(s_log, lbl)
                else:
                    loss = ce(s_log, lbl)
                loss.backward()
                opt_s.step()

        student.eval()
        preds, labels_all = [], []
        with torch.no_grad():
            for b in loader_te:
                ts, stft, wt, gaf, lbl = (b[k].to(device) for k in ['ts', 'stft', 'wt', 'gaf', 'lbl'])
                preds.extend(torch.argmax(student(ts, stft, wt, gaf), dim=1).cpu().numpy())
                labels_all.extend(lbl.cpu().numpy())

        acc, f1 = accuracy_score(labels_all, preds), f1_score(labels_all, preds, average='weighted', zero_division=0)
        results[cfg['name']] = {'acc': acc, 'f1': f1}
        print(f"[{cfg['name']}] Acc: {acc:.4f} | F1: {f1:.4f}")

    plot_ablation_bar(results, os.path.join(OUTPUT_DIR, "CWRU_Ablation_Bar.png"))
    plot_ablation_radar(results, os.path.join(OUTPUT_DIR, "CWRU_Ablation_Radar.png"))


if __name__ == "__main__":
    main()