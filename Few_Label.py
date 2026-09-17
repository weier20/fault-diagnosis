# ==============================================================================
# 半监督/少标签对比实验 (Few-Label Learning Experiment for CWRU Dataset)
# 目的：证明在标签稀缺(1%, 5%, 10%)的情况下，SSL预训练能显著提升模型鲁棒性和准确率
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
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

matplotlib.use('TkAgg')
warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.benchmark = True

# --- 目录配置 ---
DATA_DIR = r"D:\weier\test\CRWU"
OUTPUT_DIR = r"D:\weier\test\test new"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"=== Using Device: {device} ===")


# ================= 1. 数据加载与特征提取 =================
def load_cwru_data(data_dir, seq_len=1024, samples_per_class=1000):
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
            elif 'ball' in fname_lower or 'b0' in fname_lower:
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
                    # 标准化必须按样本进行，防止前向泄露
                    data.append((window - np.mean(window)) / (np.std(window) + 1e-8))
                    labels.append(label)
                    counts[label] += 1
        except:
            pass
    return np.array(data), np.array(labels)


def extract_features(signals, img_size=(32, 32)):
    print("Extracting Time-Frequency features (STFT, WT, GAF)...")
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
    # 修正了增强逻辑：在理论上应仅增强1D并重新提取2D。
    # 为了保持GPU计算的高效性，这里采用同步等比例缩放与微弱加噪模拟物理扰动
    scale = torch.rand(ts.size(0), 1, device=ts.device) * 0.4 + 0.8
    noise = 0.05
    return ts * scale + torch.randn_like(ts) * noise, stft + torch.randn_like(stft) * noise, wt + torch.randn_like(
        wt) * noise, gaf + torch.randn_like(gaf) * noise


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
    def __init__(self, hidden_dims=[512, 256, 128]):
        super().__init__()
        self.encoders = nn.ModuleList([ConvEncoder(is_1d=True)] + [ConvEncoder() for _ in range(3)])
        self.projector = nn.Sequential(nn.Linear(128 * 4, 256), nn.ReLU(), nn.Linear(256, 128))
        layers = []
        in_dim = 128 * 4
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(0.3)])
            in_dim = h
        layers.append(nn.Linear(in_dim, 4))
        self.classifier = nn.Sequential(*layers)

    def forward(self, ts, stft, wt, gaf, return_proj=False):
        fused = torch.cat([enc(x) for enc, x in zip(self.encoders, [ts, stft, wt, gaf])], dim=1)
        return self.projector(fused) if return_proj else self.classifier(fused)


class MultiModalDataset(Dataset):
    def __init__(self, ts, feats, labels):
        self.ts, self.labels = torch.FloatTensor(ts), torch.LongTensor(labels)
        self.stft, self.wt, self.gaf = (torch.FloatTensor(feats[k]) for k in ['stft', 'wavelet', 'gaf'])

    def __len__(self): return len(self.ts)

    def __getitem__(self, idx): return {'ts': self.ts[idx], 'stft': self.stft[idx], 'wt': self.wt[idx],
                                        'gaf': self.gaf[idx], 'lbl': self.labels[idx]}


# ================= 3. 核心评估流程 =================
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
    plt.figure(figsize=(8, 6))

    # 转换为百分比显示
    x_labels = [f"{int(frac * 100)}%" for frac in fractions]

    plt.plot(x_labels, acc_baseline, marker='o', linestyle='--', color='#d62728', linewidth=2,
             label='Baseline (Supervised Only, No SSL)')
    plt.plot(x_labels, acc_ssl, marker='s', linestyle='-', color='#1f77b4', linewidth=2,
             label='Proposed (SSL Pre-training + Fine-tuning)')

    plt.title('Fault Diagnosis Accuracy under Label-Scarce Conditions (CWRU Dataset)', fontsize=14, pad=15)
    plt.xlabel('Fraction of Labeled Training Data Available', fontsize=12)
    plt.ylabel('Classification Accuracy', fontsize=12)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='lower right', fontsize=11)

    # 标注数值
    for i, (b, s) in enumerate(zip(acc_baseline, acc_ssl)):
        plt.annotate(f'{b:.3f}', (i, b), textcoords="offset points", xytext=(0, -15), ha='center', fontsize=9,
                     color='#d62728')
        plt.annotate(f'{s:.3f}', (i, s), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9,
                     color='#1f77b4')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"\n[Success] Chart saved to: {save_path}")


# ================= 4. 主函数流水线 =================
def main():
    # 1. 准备数据
    X, y = load_cwru_data(DATA_DIR, seq_len=1024, samples_per_class=1000)
    feats = extract_features(X)

    # 全局 Train (70%) / Test (30%) 划分
    idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.3, random_state=42, stratify=y)

    # 测试集是固定的，用于公平评估
    dataset_te = MultiModalDataset(X[idx_te], {k: feats[k][idx_te] for k in feats}, y[idx_te])
    loader_te = DataLoader(dataset_te, batch_size=64, shuffle=False)

    # 完整训练集（用作无标签 SSL 池）
    dataset_tr_unlabeled = MultiModalDataset(X[idx_tr], {k: feats[k][idx_tr] for k in feats}, y[idx_tr])
    loader_tr_unlabeled = DataLoader(dataset_tr_unlabeled, batch_size=64, shuffle=True)

    # 设置标签稀缺比率
    label_fractions = [0.01, 0.05, 0.10, 0.20, 1.0]  # 1%, 5%, 10%, 20%, 100%

    acc_results_baseline = []
    acc_results_ssl = []

    print("\n" + "=" * 50)
    print("STARTING FEW-LABEL EXPERIMENTS")
    print("=" * 50)

    for frac in label_fractions:
        print(f"\n>>> Running Experiment with {frac * 100}% Labeled Training Data <<<")

        # 抽取带有标签的极小数据子集
        if frac == 1.0:
            idx_labeled = idx_tr
        else:
            idx_labeled, _ = train_test_split(idx_tr, train_size=frac, stratify=y[idx_tr], random_state=42)

        dataset_tr_labeled = MultiModalDataset(X[idx_labeled], {k: feats[k][idx_labeled] for k in feats},
                                               y[idx_labeled])
        loader_tr_labeled = DataLoader(dataset_tr_labeled, batch_size=32, shuffle=True)

        print(f"Total Labeled Samples Used: {len(idx_labeled)}")

        # -------------------------------------------------------------
        # 模型 A：Baseline (无 SSL，仅使用少量标签监督训练)
        # -------------------------------------------------------------
        print("  -> Training Baseline (Supervised Only)...")
        model_base = MultiModalNet().to(device)
        opt_base = torch.optim.Adam(model_base.parameters(), lr=1e-3)
        ce_loss = nn.CrossEntropyLoss()

        # 标签越少，需要的 Epoch 略微多一点以保证收敛
        epochs_ft = 10 if frac < 0.2 else 5
        for ep in range(epochs_ft):
            model_base.train()
            for b in loader_tr_labeled:
                ts, stft, wt, gaf, lbl = (b[k].to(device) for k in ['ts', 'stft', 'wt', 'gaf', 'lbl'])
                opt_base.zero_grad()
                ce_loss(model_base(ts, stft, wt, gaf), lbl).backward()
                opt_base.step()

        acc_base = evaluate_model(model_base, loader_te)
        acc_results_baseline.append(acc_base)

        # -------------------------------------------------------------
        # 模型 B：Proposed (先利用全量无标签数据 SSL，再用少量标签微调)
        # -------------------------------------------------------------
        print("  -> Training Proposed (SSL Pre-train + Fine-tune)...")
        model_ssl = MultiModalNet().to(device)

        # 阶段 1: 无标签自监督对比学习 (SSL)
        opt_pre = torch.optim.Adam(model_ssl.parameters(), lr=1e-3)
        contrastive_criterion = ContrastiveLoss(0.1)
        for ep in range(3):  # 预训练 3 轮
            model_ssl.train()
            for b in loader_tr_unlabeled:  # 使用 100% 数据，但不使用标签 (lbl)
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

        # 阶段 2: 有监督微调 (使用极少数带标签的数据)
        opt_ft = torch.optim.Adam(model_ssl.parameters(), lr=1e-3)
        for ep in range(epochs_ft):
            model_ssl.train()
            for b in loader_tr_labeled:
                ts, stft, wt, gaf, lbl = (b[k].to(device) for k in ['ts', 'stft', 'wt', 'gaf', 'lbl'])
                opt_ft.zero_grad()
                ce_loss(model_ssl(ts, stft, wt, gaf), lbl).backward()
                opt_ft.step()

        acc_ssl = evaluate_model(model_ssl, loader_te)
        acc_results_ssl.append(acc_ssl)

        print(f"  [Result] 标签比例: {frac * 100}% | Baseline ACC: {acc_base:.4f} | SSL ACC: {acc_ssl:.4f}")

    # ================= 5. 输出对比图表 =================
    plot_path = os.path.join(OUTPUT_DIR, "Few_Label_Robustness_CWRU.png")
    plot_few_label_results(label_fractions, acc_results_baseline, acc_results_ssl, plot_path)
    print("\n=== Experiment Completed Successfully! ===")


if __name__ == "__main__":
    main()