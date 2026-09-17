# ==============================================================================
# CWRU 轴承真实数据 - 11组全消融实验流水线 (包含雷达图生成)
# 最严谨的 11 组交叉对比实验，彻底验证工业场景下的多模态系统必要性
# ==============================================================================

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
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

matplotlib.use('TkAgg')
warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)

DATA_DIR = r"D:\weier\test\CRWU"
OUTPUT_DIR = r"D:\weier\test\test new"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# ================= 1. CWRU 数据加载与提取 =================
def load_cwru_data(data_dir, seq_len=1024, samples_per_class=100):
    print(f"Scanning CWRU dataset in: {data_dir} ...")
    data, labels = [], []
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    mat_files = glob.glob(os.path.join(data_dir, '**', '*.mat'), recursive=True)

    for file_path in mat_files:
        filename = os.path.basename(file_path)
        label = -1
        match = re.search(r'(\d+)', filename)
        if match:
            num = int(match.group(1))
            if num in [97, 98, 99, 100]:
                label = 0
            elif num in [105, 106, 107, 108, 169, 170, 171, 172, 209, 210, 211, 212, 274, 275, 276, 277]:
                label = 1
            elif num in [130, 131, 132, 133, 144, 145, 146, 147, 156, 158, 159, 160, 197, 198, 199, 200, 234, 235, 236,
                         237, 246, 247, 248, 249, 250, 258, 259, 260, 261, 262, 294, 295, 296, 297, 298, 309, 310, 311,
                         312, 313, 315, 316, 317, 318]:
                label = 2
            elif num in [118, 119, 120, 121, 185, 186, 187, 188, 222, 223, 224, 225, 282, 283, 284, 285]:
                label = 3

        if label == -1:
            fname_lower = filename.lower()
            if 'normal' in fname_lower or 'base' in fname_lower:
                label = 0
            elif 'ir' in fname_lower or 'inner' in fname_lower:
                label = 1
            elif 'or' in fname_lower or 'outer' in fname_lower:
                label = 2
            elif 'ball' in fname_lower or 'b0' in fname_lower or 'b1' in fname_lower or 'b2' in fname_lower:
                label = 3

        if label == -1 or counts[label] >= samples_per_class: continue
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
                for i in range(len(sig) // seq_len):
                    if counts[label] >= samples_per_class: break
                    window = sig[i * seq_len: (i + 1) * seq_len]
                    data.append((window - np.mean(window)) / (np.std(window) + 1e-8))
                    labels.append(label)
                    counts[label] += 1
        except:
            pass
    return np.array(data), np.array(labels)


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


# ================= 2. 网络组件 =================
class ContrastiveLoss(nn.Module):
    def __init__(self, temp=0.1):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=2)

    def forward(self, proj):
        bs = proj.size(0) // 2
        sim = self.cos(proj.unsqueeze(1), proj.unsqueeze(0)) / self.temp
        loss = 0
        for i in range(2 * bs):
            pos_idx = (i + bs) % (2 * bs) if i < bs else i - bs
            pos_sim = sim[i, pos_idx]
            mask = torch.ones(2 * bs, dtype=torch.bool, device=proj.device)
            mask[i], mask[pos_idx] = False, False
            neg_sims = sim[i, mask]
            loss += -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.sum(torch.exp(neg_sims))))
        return loss / (2 * bs)


def augment_batch(ts, stft, wt, gaf):
    scale = torch.rand(ts.size(0), 1, device=ts.device) * 0.4 + 0.8
    return ts * scale + torch.randn_like(ts) * 0.05, stft + torch.randn_like(stft) * 0.05, wt + torch.randn_like(
        wt) * 0.05, gaf + torch.randn_like(gaf) * 0.05


class ConvEncoder(nn.Module):
    def __init__(self, is_1d=False):
        super().__init__()
        self.is_1d = is_1d
        if is_1d:
            self.net = nn.Sequential(nn.Conv1d(1, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
                                     nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(16),
                                     nn.Flatten(), nn.Linear(64 * 16, 128))
        else:
            self.net = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                     nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)),
                                     nn.Flatten(), nn.Linear(64 * 16, 128))

    def forward(self, x):
        if self.is_1d and x.dim() == 2: x = x.unsqueeze(1)
        if not self.is_1d:
            x = x.unsqueeze(0).unsqueeze(0) if x.dim() == 2 else x.unsqueeze(1)
            if x.size(2) < 4: x = F.interpolate(x, size=(8, 8), mode='bilinear')
        return self.net(x)


class MultiModalNet(nn.Module):
    def __init__(self, encoders, hidden_dims, active_mods=[True, True, True, True]):
        super().__init__()
        self.encoders = nn.ModuleList(encoders)
        self.active_mods = active_mods
        in_dim = 128 * sum(active_mods)
        self.projector = nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(), nn.Linear(256, 128))
        layers = []
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(0.3)])
            in_dim = h
        layers.append(nn.Linear(in_dim, 4))
        self.classifier = nn.Sequential(*layers)

    def forward(self, ts, stft, wt, gaf, return_proj=False):
        inputs = [ts, stft, wt, gaf]
        fused = [enc(x) for i, (enc, x) in enumerate(zip(self.encoders, inputs)) if self.active_mods[i]]
        fused = torch.cat(fused, dim=1)
        return self.projector(fused) if return_proj else self.classifier(fused)


class MultiModalDataset(Dataset):
    def __init__(self, ts, feats, labels):
        self.ts, self.labels = torch.FloatTensor(ts), torch.LongTensor(labels)
        self.stft, self.wt, self.gaf = (torch.FloatTensor(feats[k]) for k in ['stft', 'wavelet', 'gaf'])

    def __len__(self): return len(self.ts)

    def __getitem__(self, idx): return {'ts': self.ts[idx], 'stft': self.stft[idx], 'wt': self.wt[idx],
                                        'gaf': self.gaf[idx], 'lbl': self.labels[idx]}


# ================= 3. 消融图表绘制 (柱状图 + 雷达图) =================
def plot_ablation_bar(results, save_path):
    names = list(results.keys())
    accs = [results[n]['acc'] for n in names]
    f1s = [results[n]['f1'] for n in names]
    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    b1 = ax.bar(x - width / 2, accs, width, label='Accuracy', color='#4c72b0')
    b2 = ax.bar(x + width / 2, f1s, width, label='F1 Score', color='#55a868')

    ax.set_ylabel('Scores', fontsize=12)
    ax.set_title('CWRU Data: Complete Ablation Study (11 Configurations)', fontsize=15, pad=15)
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
    """【新增】雷达图生成函数"""
    labels = list(results.keys())
    num_vars = len(labels)

    accs = [results[n]['acc'] for n in labels]
    f1s = [results[n]['f1'] for n in labels]

    # 闭环追加
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    accs += accs[:1]
    f1s += f1s[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # 去掉最前面的序号（如 "1. "），让图表更清爽
    short_labels = [l.split('. ', 1)[-1] for l in labels]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(short_labels, fontsize=10)

    ax.set_ylim(0, 1.05)

    ax.plot(angles, accs, linewidth=2, linestyle='solid', label='Accuracy', color='#4c72b0')
    ax.fill(angles, accs, alpha=0.2, color='#4c72b0')

    ax.plot(angles, f1s, linewidth=2, linestyle='solid', label='F1 Score', color='#55a868')
    ax.fill(angles, f1s, alpha=0.2, color='#55a868')

    plt.title('Ablation Study Radar Chart', size=16, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1))
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# ================= 4. 主干流水线 =================
def main():
    X, y = load_cwru_data(DATA_DIR, seq_len=1024, samples_per_class=120)
    feats = extract_features(X)
    idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.3, random_state=42, stratify=y)

    loader_tr = DataLoader(MultiModalDataset(X[idx_tr], {k: feats[k][idx_tr] for k in feats}, y[idx_tr]), batch_size=32,
                           shuffle=True)
    loader_te = DataLoader(MultiModalDataset(X[idx_te], {k: feats[k][idx_te] for k in feats}, y[idx_te]), batch_size=32,
                           shuffle=False)

    configs = [
        {"name": "1. Full System", "mods": [True, True, True, True], "ssl": True, "kd": True},
        {"name": "2. w/o KD (No Distillation)", "mods": [True, True, True, True], "ssl": True, "kd": False},
        {"name": "3. w/o SSL (No Pretrain)", "mods": [True, True, True, True], "ssl": False, "kd": True},
        {"name": "4. Only 1D TS", "mods": [True, False, False, False], "ssl": True, "kd": True},
        {"name": "5. Only STFT", "mods": [False, True, False, False], "ssl": True, "kd": True},
        {"name": "6. Only WT", "mods": [False, False, True, False], "ssl": True, "kd": True},
        {"name": "7. Only GAF", "mods": [False, False, False, True], "ssl": True, "kd": True},
        {"name": "8. w/o 1D TS (Missing TS)", "mods": [False, True, True, True], "ssl": True, "kd": True},
        {"name": "9. w/o STFT (Missing STFT)", "mods": [True, False, True, True], "ssl": True, "kd": True},
        {"name": "10. w/o WT (Missing WT)", "mods": [True, True, False, True], "ssl": True, "kd": True},
        {"name": "11. w/o GAF (Missing GAF)", "mods": [True, True, True, False], "ssl": True, "kd": True}
    ]

    results = {}
    for cfg in configs:
        print(f"\n>>> Running Ablation: {cfg['name']} <<<")

        def _build(h):
            return MultiModalNet([ConvEncoder(is_1d=True).to(device)] + [ConvEncoder().to(device) for _ in range(3)], h,
                                 cfg['mods']).to(device)

        teacher, student = _build([512, 256, 128]), _build([256, 128])

        if cfg['ssl']:
            opt_pre, crit = torch.optim.Adam(teacher.parameters(), lr=1e-3), ContrastiveLoss(0.1)
            for _ in range(4):
                teacher.train()
                for b in loader_tr:
                    ts, stft, wt, gaf = (b[k].to(device) for k in ['ts', 'stft', 'wt', 'gaf'])
                    ts1, stft1, wt1, gaf1 = augment_batch(ts, stft, wt, gaf)
                    ts2, stft2, wt2, gaf2 = augment_batch(ts, stft, wt, gaf)
                    loss = crit(
                        torch.cat([teacher(ts1, stft1, wt1, gaf1, True), teacher(ts2, stft2, wt2, gaf2, True)], dim=0))
                    opt_pre.zero_grad();
                    loss.backward();
                    opt_pre.step()

        if cfg['kd']:
            opt_t, ce = torch.optim.Adam(teacher.parameters(), lr=1e-3), nn.CrossEntropyLoss()
            for _ in range(5):
                teacher.train()
                for b in loader_tr:
                    ts, stft, wt, gaf, lbl = (b[k].to(device) for k in ['ts', 'stft', 'wt', 'gaf', 'lbl'])
                    opt_t.zero_grad();
                    ce(teacher(ts, stft, wt, gaf), lbl).backward();
                    opt_t.step()

        opt_s, ce, kl = torch.optim.Adam(student.parameters(), lr=1e-3), nn.CrossEntropyLoss(), nn.KLDivLoss(
            reduction='batchmean')
        for _ in range(12):
            student.train()
            for b in loader_tr:
                ts, stft, wt, gaf, lbl = (b[k].to(device) for k in ['ts', 'stft', 'wt', 'gaf', 'lbl'])
                opt_s.zero_grad()
                s_log = student(ts, stft, wt, gaf)
                if cfg['kd']:
                    with torch.no_grad():
                        t_log = teacher(ts, stft, wt, gaf)
                    loss = 0.6 * (
                                kl(F.log_softmax(s_log / 3.0, dim=1), F.softmax(t_log / 3.0, dim=1)) * 9.0) + 0.4 * ce(
                        s_log, lbl)
                else:
                    loss = ce(s_log, lbl)
                loss.backward();
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

    # 生成两种图表
    plot_ablation_bar(results, os.path.join(OUTPUT_DIR, "CWRU_Ablation_Bar.png"))
    plot_ablation_radar(results, os.path.join(OUTPUT_DIR, "CWRU_Ablation_Radar.png"))
    print(f"\n=== Ablation Study Completed! Charts saved to: {OUTPUT_DIR} ===")


if __name__ == "__main__":
    main()