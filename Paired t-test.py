import os
import time
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
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

matplotlib.use('TkAgg')
warnings.filterwarnings('ignore')

# --- 配置 ---
DATA_DIR = r"D:\weier\test\CRWU"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"=== Device: {device} ===")


# 数据准备
def load_cwru_data(data_dir, seq_len=1024, samples_per_class=500):
    if not os.path.exists(data_dir): raise FileNotFoundError(f"Directory not found: {data_dir}")
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
                    sig = mat_dict[key].flatten();
                    break
            if sig is None:
                for key in mat_dict.keys():
                    if not key.startswith('__') and isinstance(mat_dict[key], np.ndarray):
                        sig = mat_dict[key].flatten();
                        break
            if sig is not None:
                for i in range(len(sig) // seq_len):
                    if counts[label] >= samples_per_class: break
                    window_sig = sig[i * seq_len: (i + 1) * seq_len]
                    window_sig = (window_sig - np.mean(window_sig)) / (np.std(window_sig) + 1e-8)
                    data.append(window_sig);
                    labels.append(label);
                    counts[label] += 1
        except Exception:
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


class MultiModalDataset(Dataset):
    def __init__(self, ts, feats, labels=None):
        self.ts = torch.FloatTensor(ts)
        self.labels = torch.LongTensor(labels) if labels is not None else torch.zeros(len(ts), dtype=torch.long)
        self.stft, self.wt, self.gaf = (torch.FloatTensor(feats[k]) for k in ['stft', 'wavelet', 'gaf'])

    def __len__(self): return len(self.ts)

    def __getitem__(self, idx):
        return {'ts': self.ts[idx], 'stft': self.stft[idx], 'wt': self.wt[idx], 'gaf': self.gaf[idx],
                'lbl': self.labels[idx]}


# 核心模块与网络
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
    except ValueError:
        auc_val = 0.5
    return acc, auc_val


# 主实验
def run_few_shot_experiment():
    print("=== EXTRACTING DATA ===")
    X, y = load_cwru_data(DATA_DIR, seq_len=1024, samples_per_class=500)  # 总计 2000 个样本
    feats = extract_features(X)

    # 我们设定每类只有 15 个标注样本，共 60 个
    N_SHOTS_PER_CLASS = 15
    SEEDS = [42, 1024, 2024, 2026, 3407]

    acc_baseline, auc_baseline = [], []
    acc_proposed, auc_proposed = [], []

    for i, seed in enumerate(SEEDS):
        print(f"\n=======================================================")
        print(f"=== Run {i + 1}/5 | Random Seed: {seed} | Few-Shot ({N_SHOTS_PER_CLASS}/class) ===")
        print(f"=======================================================")

        torch.manual_seed(seed)
        np.random.seed(seed)

        # 1. 划分训练集(70%) 和 测试集(30%)
        idx_tr_full, idx_te = train_test_split(np.arange(len(y)), test_size=0.3, stratify=y, random_state=seed)

        # 2. 从训练集中提取极小样本作为 "有标签数据" (Labeled Data)
        sss = StratifiedShuffleSplit(n_splits=1, train_size=N_SHOTS_PER_CLASS * 4, random_state=seed)
        for idx_few_shot, _ in sss.split(idx_tr_full, y[idx_tr_full]):
            idx_tr_labeled = idx_tr_full[idx_few_shot]

        print(
            f"Data Split -> Total Unlabeled Train Pool: {len(idx_tr_full)} | Labeled Few-Shot Train: {len(idx_tr_labeled)} | Test: {len(idx_te)}")

        # DataLoaders
        batch_sz = 16
        loader_unlabeled = DataLoader(
            MultiModalDataset(X[idx_tr_full], {k: feats[k][idx_tr_full] for k in feats}, y[idx_tr_full]), batch_size=64,
            shuffle=True)
        loader_labeled = DataLoader(
            MultiModalDataset(X[idx_tr_labeled], {k: feats[k][idx_tr_labeled] for k in feats}, y[idx_tr_labeled]),
            batch_size=batch_sz, shuffle=True)
        loader_test = DataLoader(MultiModalDataset(X[idx_te], {k: feats[k][idx_te] for k in feats}, y[idx_te]),
                                 batch_size=64, shuffle=False)

        ce_loss = nn.CrossEntropyLoss()


        print("\n[1/3] Training Baseline on Few-Shot Labeled Data...")
        base_model = MultiModalNet([256, 128]).to(device)
        opt_base = torch.optim.Adam(base_model.parameters(), lr=1e-3)

        for ep in range(30):
            base_model.train()
            for b in loader_labeled:
                ts, stft, wt, gaf, lbl = (b[k].to(device) for k in ['ts', 'stft', 'wt', 'gaf', 'lbl'])
                opt_base.zero_grad()
                ce_loss(base_model(ts, stft, wt, gaf), lbl).backward()
                opt_base.step()

        b_acc, b_auc = evaluate(base_model, loader_test)
        acc_baseline.append(b_acc);
        auc_baseline.append(b_auc)
        print(f"-> Baseline Acc: {b_acc:.4f} | AUC: {b_auc:.4f}")

        print("\n[2/3] Training Proposed Teacher (SSL on Unlabeled -> FT on Labeled)...")
        teacher = MultiModalNet([512, 256, 128]).to(device)

        # Stage 1: SSL 预训练
        opt_pre = torch.optim.Adam(teacher.parameters(), lr=1e-3)
        ssl_loss_fn = ContrastiveLoss(temperature=0.1)
        for ep in range(10):  # 预训练10轮
            teacher.train()
            for b in loader_unlabeled:
                ts, stft, wt, gaf = (b[k].to(device) for k in ['ts', 'stft', 'wt', 'gaf'])
                ts1, stft1, wt1, gaf1 = augment_batch(ts, stft, wt, gaf)
                ts2, stft2, wt2, gaf2 = augment_batch(ts, stft, wt, gaf)
                loss = ssl_loss_fn(
                    torch.cat([teacher(ts1, stft1, wt1, gaf1, True), teacher(ts2, stft2, wt2, gaf2, True)], dim=0))
                opt_pre.zero_grad();
                loss.backward();
                opt_pre.step()

        # Stage 2: 有监督微调
        opt_ft = torch.optim.Adam(teacher.parameters(), lr=1e-3)
        for ep in range(20):
            teacher.train()
            for b in loader_labeled:
                ts, stft, wt, gaf, lbl = (b[k].to(device) for k in ['ts', 'stft', 'wt', 'gaf', 'lbl'])
                opt_ft.zero_grad();
                ce_loss(teacher(ts, stft, wt, gaf), lbl).backward();
                opt_ft.step()

        print("[3/3] Training Proposed Student (KD on Labeled Data)...")
        student = MultiModalNet([256, 128]).to(device)
        opt_kd = torch.optim.Adam(student.parameters(), lr=1e-3)
        kl_loss = nn.KLDivLoss(reduction='batchmean')

        T, alpha = 3.0, 0.7  # 恢复正常蒸馏参数
        for ep in range(30):
            student.train()
            for b in loader_labeled:
                ts, stft, wt, gaf, lbl = (b[k].to(device) for k in ['ts', 'stft', 'wt', 'gaf', 'lbl'])
                opt_kd.zero_grad()
                s_log = student(ts, stft, wt, gaf)
                with torch.no_grad(): t_log = teacher(ts, stft, wt, gaf)
                loss = alpha * (kl_loss(F.log_softmax(s_log / T, dim=1), F.softmax(t_log / T, dim=1)) * (T ** 2)) + (
                            1 - alpha) * ce_loss(s_log, lbl)
                loss.backward();
                opt_kd.step()

        p_acc, p_auc = evaluate(student, loader_test)
        acc_proposed.append(p_acc);
        auc_proposed.append(p_auc)
        print(f"-> Proposed KD Acc: {p_acc:.4f} | AUC: {p_auc:.4f}")

    # 统计显著性报告
    print("\n" + "=" * 60)
    print("=== FINAL PAPER REPORT: EXTREME FEW-SHOT LEARNING ===")
    print("=" * 60)

    mean_b_acc, std_b_acc = np.mean(acc_baseline), np.std(acc_baseline)
    mean_p_acc, std_p_acc = np.mean(acc_proposed), np.std(acc_proposed)
    mean_b_auc, std_b_auc = np.mean(auc_baseline), np.std(auc_baseline)
    mean_p_auc, std_p_auc = np.mean(auc_proposed), np.std(auc_proposed)

    t_acc, p_acc_val = stats.ttest_rel(acc_proposed, acc_baseline)
    t_auc, p_auc_val = stats.ttest_rel(auc_proposed, auc_baseline)

    p_acc_1tail = p_acc_val / 2 if t_acc > 0 else 1.0
    p_auc_1tail = p_auc_val / 2 if t_auc > 0 else 1.0

    print(f"[1] Accuracy:")
    print(f"    Baseline (From Scratch): {mean_b_acc * 100:.2f}% ± {std_b_acc * 100:.2f}%")
    print(f"    Proposed (SSL + KD):     {mean_p_acc * 100:.2f}% ± {std_p_acc * 100:.2f}%")
    print(f"    Paired T-test P-value:   {p_acc_1tail:.4e}")

    print(f"\n[2] Macro-AUC:")
    print(f"    Baseline (From Scratch): {mean_b_auc:.4f} ± {std_b_auc:.4f}")
    print(f"    Proposed (SSL + KD):     {mean_p_auc:.4f} ± {std_p_auc:.4f}")
    print(f"    Paired T-test P-value:   {p_auc_1tail:.4e}")

    if p_acc_1tail < 0.01:
        print("\n>>> 结论: P < 0.01")
    else:
        print("\n>>> 请检查结果。通常小样本测试能轻易拉开巨大的差距。")
    print("=" * 60)


if __name__ == "__main__":
    run_few_shot_experiment()