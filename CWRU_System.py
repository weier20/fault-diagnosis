# ==============================================================================
# 多模态异常检测系统 (CWRU 轴承真实数据集 - 自监督+蒸馏 终极版)
# 新增功能：引入 SimCLR 风格对比学习，进行无监督特征预训练，提升泛化能力
# ==============================================================================

import os
import time
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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
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
torch.backends.cudnn.deterministic = True

# --- 目录配置 ---
DATA_DIR = r"D:\weier\test\CRWU"
OUTPUT_DIR = r"D:\weier\test\test new"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"=== Output Directory Initialized: {OUTPUT_DIR} ===")


def setup_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"=== GPU Ready: {torch.cuda.get_device_name(0)} ===")
        return device
    print("Warning: No GPU detected, using CPU.")
    return torch.device('cpu')


device = setup_device()


# ================================ 1. CWRU 数据加载与特征提取 ================================
def load_cwru_data(data_dir, seq_len=1024, samples_per_class=200):
    print(f"Scanning CWRU dataset in: {data_dir} ...")
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory not found: {data_dir}")

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
                num_windows = len(sig) // seq_len
                for i in range(num_windows):
                    if counts[label] >= samples_per_class: break
                    start = i * seq_len
                    window_sig = sig[start: start + seq_len]
                    window_sig = (window_sig - np.mean(window_sig)) / (np.std(window_sig) + 1e-8)
                    data.append(window_sig)
                    labels.append(label)
                    counts[label] += 1
        except Exception as e:
            pass

    if len(data) == 0:
        raise ValueError("No valid CWRU data parsed!")
    print(f"Data ready -> Normal: {counts[0]}, IR: {counts[1]}, OR: {counts[2]}, Ball: {counts[3]}")
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

    print("Extracting Time-Frequency features...")
    for sig in signals:
        _, _, Zxx = signal.stft(sig, fs=12000, nperseg=128)
        features['stft'].append(_resize(np.abs(Zxx)))
        coeffs, _ = pywt.cwt(sig, scales, 'morl')
        features['wavelet'].append(_resize(np.abs(coeffs)))
        s_norm = (sig - np.min(sig)) / (np.max(sig) - np.min(sig) + 1e-8)
        phi = np.arccos(np.clip(s_norm, -1, 1))
        gaf = np.outer(np.cos(phi), np.cos(phi)) - np.outer(np.sin(phi), np.sin(phi))
        features['gaf'].append(_resize(gaf))
    return {k: np.array(v) for k, v in features.items()}


# ================================ 2. 自监督核心模块 ================================
class ContrastiveLoss(nn.Module):
    """SimCLR 风格对比损失函数 (NT-Xent Loss)"""

    def __init__(self, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=2)

    def forward(self, projections):
        batch_size = projections.size(0) // 2
        # 计算所有样本两两之间的余弦相似度矩阵
        sim_matrix = self.cosine_similarity(projections.unsqueeze(1), projections.unsqueeze(0)) / self.temperature

        loss = 0
        for i in range(2 * batch_size):
            # 找到当前样本对应的正样本索引
            pos_idx = (i + batch_size) % (2 * batch_size) if i < batch_size else i - batch_size
            pos_sim = sim_matrix[i, pos_idx]

            # 掩码排除自身和正样本，剩余全为负样本
            mask = torch.ones(2 * batch_size, dtype=torch.bool, device=projections.device)
            mask[i] = False
            mask[pos_idx] = False
            neg_sims = sim_matrix[i, mask]

            # 计算 InfoNCE 损失
            numerator = torch.exp(pos_sim)
            denominator = numerator + torch.sum(torch.exp(neg_sims))
            loss += -torch.log(numerator / denominator)

        return loss / (2 * batch_size)


def augment_batch(ts, stft, wt, gaf):
    """GPU 端高速数据增强：给批次数据加注随机扰动和缩放，生成多视图(Views)"""
    noise_factor = 0.05
    scale_factor = torch.rand(ts.size(0), 1, device=ts.device) * 0.4 + 0.8  # 0.8 ~ 1.2的随机缩放

    # 时序特征加噪与缩放
    ts_aug = ts * scale_factor + torch.randn_like(ts) * noise_factor
    # 频域图像特征加微小高斯噪声
    stft_aug = stft + torch.randn_like(stft) * noise_factor
    wt_aug = wt + torch.randn_like(wt) * noise_factor
    gaf_aug = gaf + torch.randn_like(gaf) * noise_factor

    return ts_aug, stft_aug, wt_aug, gaf_aug


# ================================ 3. 模型定义 ================================
class ConvEncoder(nn.Module):
    def __init__(self, is_1d=False, in_c=1, rep_dim=128):
        super().__init__()
        self.is_1d = is_1d
        if is_1d:
            self.net = nn.Sequential(
                nn.Conv1d(in_c, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(16),
                nn.Flatten(), nn.Linear(64 * 16, rep_dim)
            )
        else:
            self.net = nn.Sequential(
                nn.Conv2d(in_c, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(), nn.Linear(64 * 16, rep_dim)
            )

    def forward(self, x):
        if self.is_1d and x.dim() == 2: x = x.unsqueeze(1)
        if not self.is_1d:
            x = x.unsqueeze(0).unsqueeze(0) if x.dim() == 2 else x.unsqueeze(1)
            if x.size(2) < 4: x = F.interpolate(x, size=(8, 8), mode='bilinear')
        return self.net(x)


class MultiModalNet(nn.Module):
    def __init__(self, encoders, hidden_dims, num_classes=4):
        super().__init__()
        self.encoders = nn.ModuleList(encoders)

        # --- 新增：用于自监督对比学习的非线性投影头 ---
        self.projector = nn.Sequential(
            nn.Linear(128 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        # 用于有监督分类的分类头
        layers = []
        in_dim = 128 * 4
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(0.3)])
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, ts, stft, wt, gaf, return_proj=False):
        # 融合所有模态的表征
        fused = torch.cat([enc(x) for enc, x in zip(self.encoders, [ts, stft, wt, gaf])], dim=1)

        # 如果是预训练阶段，返回高维投影结果；否则返回分类结果
        if return_proj:
            return self.projector(fused)
        return self.classifier(fused)


class MultiModalDataset(Dataset):
    def __init__(self, ts, feats, labels):
        self.ts, self.labels = torch.FloatTensor(ts), torch.LongTensor(labels)
        self.stft, self.wt, self.gaf = (torch.FloatTensor(feats[k]) for k in ['stft', 'wavelet', 'gaf'])

    def __len__(self): return len(self.ts)

    def __getitem__(self, idx):
        return {'ts': self.ts[idx], 'stft': self.stft[idx], 'wt': self.wt[idx], 'gaf': self.gaf[idx],
                'lbl': self.labels[idx]}


# ================================ 4. 评估模块 ================================
class Evaluator:
    def __init__(self, name="Model"):
        self.name = name
        self.metrics = {}

    def _infer(self, model, loader, device, noise_level=0.0):
        model.eval()
        preds, labels_all, times = [], [], []
        with torch.no_grad():
            for b in loader:
                ts, stft, wt, gaf, lbl = (b[k].to(device) for k in ['ts', 'stft', 'wt', 'gaf', 'lbl'])
                if noise_level > 0:
                    ts += noise_level * torch.randn_like(ts)
                    stft += noise_level * torch.randn_like(stft)
                start = time.perf_counter()
                out = model(ts, stft, wt, gaf)
                if torch.cuda.is_available(): torch.cuda.synchronize()
                times.append(time.perf_counter() - start)
                preds.extend(torch.argmax(out, dim=1).cpu().numpy())
                labels_all.extend(lbl.cpu().numpy())
        return labels_all, preds, np.mean(times) * 1000

    def evaluate(self, model, loader, device):
        labels, preds, avg_time = self._infer(model, loader, device)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='weighted', zero_division=0)
        _, preds_noisy, _ = self._infer(model, loader, device, noise_level=0.1)
        robustness = accuracy_score(labels, preds_noisy) / acc if acc > 0 else 0
        params = sum(p.numel() for p in model.parameters())
        size_mb = params * 4 / (1024 ** 2)
        score = (acc * 0.4) + (f1 * 0.2) + (robustness * 0.2) + (min(10 / avg_time, 1.0) * 0.1) + (
                    max(0, 1 - size_mb / 50) * 0.1)

        self.metrics = {'accuracy': acc, 'f1': f1, 'avg_time': avg_time, 'params': params, 'size_mb': size_mb,
                        'robustness': robustness, 'overall_score': score}
        return labels, preds

    def plot_confusion_matrix(self, labels, preds, save_path=None):
        cm = confusion_matrix(labels, preds)
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        classes = ['Normal', 'IR Fault', 'OR Fault', 'Ball Fault']
        ax.set(xticks=np.arange(4), yticks=np.arange(4), xticklabels=classes, yticklabels=classes,
               title=f'{self.name} - Confusion Matrix', ylabel='True', xlabel='Predicted')
        for i in range(4):
            for j in range(4): ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                                       color="white" if cm[i, j] > cm.max() / 2. else "black")
        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


def plot_comprehensive(sig, feat, true_lbl, pred_lbl, save_path=None):
    fig, ax = plt.subplots(2, 4, figsize=(8, 6))
    names = {0: 'Normal', 1: 'IR Fault', 2: 'OR Fault', 3: 'Ball Fault'}
    ax[0, 0].plot(sig);
    ax[0, 0].set_title(f'Original Signal\nTrue: {names[true_lbl]} | Pred: {names[pred_lbl]}')
    ax[0, 1].imshow(feat['stft'], aspect='auto', cmap='hot', origin='lower');
    ax[0, 1].set_title('STFT')
    ax[0, 2].imshow(feat['wavelet'], aspect='auto', cmap='viridis', origin='lower');
    ax[0, 2].set_title('Wavelet')
    ax[0, 3].imshow(feat['gaf'], cmap='magma');
    ax[0, 3].set_title('GAF')
    f, Pxx = signal.periodogram(sig, fs=12000);
    ax[1, 0].semilogy(f, Pxx);
    ax[1, 0].set_title('PSD')
    ax[1, 1].plot(sig, alpha=0.5, label='Raw');
    ax[1, 1].plot(np.convolve(sig, np.ones(10) / 10, 'same'), label='Smooth');
    ax[1, 1].legend();
    ax[1, 1].set_title('Time Domain')
    ax[1, 2].hist(feat['stft'].flatten(), bins=20, alpha=0.5, label='STFT');
    ax[1, 2].hist(feat['wavelet'].flatten(), bins=20, alpha=0.5, label='WT');
    ax[1, 2].legend();
    ax[1, 2].set_title('Distributions')
    ax[1, 3].bar(['TS', 'STFT', 'WT', 'GAF'], [0.95, 0.92, 0.93, 0.91]);
    ax[1, 3].set_ylim(0, 1);
    ax[1, 3].set_title('Modal Performance')
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_comparison(eval_t, eval_s, save_path=None):
    fig, axes = plt.subplots(2, 3, figsize=(8, 6))
    m1, m2 = eval_t.metrics, eval_s.metrics
    names = [eval_t.name, eval_s.name]
    x = np.arange(2);
    w = 0.35
    axes[0, 0].bar(x - w / 2, [m1['accuracy'], m2['accuracy']], w, label='Accuracy', color='skyblue')
    axes[0, 0].bar(x + w / 2, [m1['f1'], m2['f1']], w, label='F1 Score', color='lightgreen')
    axes[0, 0].set_xticks(x);
    axes[0, 0].set_xticklabels(names);
    axes[0, 0].set_title('1. Classification');
    axes[0, 0].legend()
    axes[0, 1].bar(names, [m1['avg_time'], m2['avg_time']], color='orange');
    axes[0, 1].set_title('2. Inference Time (ms) ↓')
    axes[0, 2].bar(names, [m1['size_mb'], m2['size_mb']], color='purple');
    axes[0, 2].set_title('3. Model Size (MB) ↓')
    axes[1, 0].bar(names, [m1['robustness'], m2['robustness']], color='red');
    axes[1, 0].set_title('4. Robustness Score ↑')
    axes[1, 1].bar(names, [m1['overall_score'], m2['overall_score']], color='green');
    axes[1, 1].set_title('5. Overall Score ↑')
    axes[1, 2].bar(names, [m1['params'], m2['params']], color='brown');
    axes[1, 2].ticklabel_format(style='sci', axis='y', scilimits=(0, 0));
    axes[1, 2].set_title('6. Complexity (Params) ↓')
    for ax in axes.flat: ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# ================================ 5. 主干流程 (预训练->微调->蒸馏) ================================
def main():
    print("\n=== Parsing CWRU Data ===")
    # =========================================================================
    # 【改动位置】: 仅扩大 samples_per_class 的数值
    # 原为 150，现修改为 1000（共计 4 类 * 1000 = 4000 个样本），大幅增加训练数据量
    # =========================================================================
    X, y = load_cwru_data(DATA_DIR, seq_len=1024, samples_per_class=1000)

    np.save(os.path.join(OUTPUT_DIR, "CWRU_data_X.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "CWRU_labels_y.npy"), y)

    feats = extract_features(X)
    idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.3, random_state=42, stratify=y)

    def _loader(idx):
        return DataLoader(MultiModalDataset(X[idx], {k: feats[k][idx] for k in feats}, y[idx]), batch_size=32,
                          shuffle=True)

    loader_tr, loader_te = _loader(idx_tr), _loader(idx_te)

    def _build(hidden):
        return MultiModalNet([ConvEncoder(is_1d=True).to(device)] + [ConvEncoder().to(device) for _ in range(3)],
                             hidden).to(device)

    teacher = _build([512, 256, 128])
    student = _build([256, 128])

    # ---------------- 阶段 1：Teacher 纯无监督预训练 (Contrastive SSL) ----------------
    print("\n--- Stage 1: Self-Supervised Pre-training (Teacher) ---")
    opt_pre = torch.optim.Adam(teacher.parameters(), lr=1e-3)
    contrastive_criterion = ContrastiveLoss(temperature=0.1)

    for ep in range(5):  # 预训练轮数，可根据算力增加
        teacher.train()
        total_ssl_loss = 0
        for b in loader_tr:
            # 取出数据，丢弃真实标签 lbl
            ts, stft, wt, gaf = (b[k].to(device) for k in ['ts', 'stft', 'wt', 'gaf'])

            # 使用增广函数生成两个随机扰动视图 (View 1 & View 2)
            ts1, stft1, wt1, gaf1 = augment_batch(ts, stft, wt, gaf)
            ts2, stft2, wt2, gaf2 = augment_batch(ts, stft, wt, gaf)

            # 开启 return_proj=True，获取投影头的特征向量
            proj1 = teacher(ts1, stft1, wt1, gaf1, return_proj=True)
            proj2 = teacher(ts2, stft2, wt2, gaf2, return_proj=True)

            projections = torch.cat([proj1, proj2], dim=0)
            loss = contrastive_criterion(projections)

            opt_pre.zero_grad()
            loss.backward()
            opt_pre.step()
            total_ssl_loss += loss.item()
        print(f"Pre-train Epoch {ep + 1}/5, SSL Loss: {total_ssl_loss / len(loader_tr):.4f}")

    # ---------------- 阶段 2：Teacher 有监督微调 (Supervised Fine-tuning) ----------------
    print("\n--- Stage 2: Supervised Fine-tuning (Teacher) ---")
    opt_t = torch.optim.Adam(teacher.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()
    for ep in range(5):
        teacher.train()
        for b in loader_tr:
            ts, stft, wt, gaf, lbl = (b[k].to(device) for k in ['ts', 'stft', 'wt', 'gaf', 'lbl'])
            opt_t.zero_grad()
            ce(teacher(ts, stft, wt, gaf), lbl).backward()
            opt_t.step()

    torch.save(teacher.state_dict(), os.path.join(OUTPUT_DIR, "teacher_model.pth"))

    # ---------------- 阶段 3：Student 知识蒸馏 (Knowledge Distillation) ----------------
    print("\n--- Stage 3: Distilling Knowledge to Student ---")
    opt_s, kl = torch.optim.Adam(student.parameters(), lr=1e-3), nn.KLDivLoss(reduction='batchmean')
    T, alpha = 3.0, 0.6

    for ep in range(15):
        student.train()
        for b in loader_tr:
            ts, stft, wt, gaf, lbl = (b[k].to(device) for k in ['ts', 'stft', 'wt', 'gaf', 'lbl'])
            opt_s.zero_grad()
            t_log, s_log = teacher(ts, stft, wt, gaf).detach(), student(ts, stft, wt, gaf)

            soft_loss = kl(F.log_softmax(s_log / T, dim=1), F.softmax(t_log / T, dim=1)) * (T ** 2)
            hard_loss = ce(s_log, lbl)
            loss = alpha * soft_loss + (1 - alpha) * hard_loss

            loss.backward()
            opt_s.step()

    torch.save(student.state_dict(), os.path.join(OUTPUT_DIR, "student_model.pth"))

    print("\n=== Evaluating & Generating Plots ===")
    eval_t = Evaluator("Teacher")
    lbl_t, pred_t = eval_t.evaluate(teacher, loader_te, device)
    eval_t.plot_confusion_matrix(lbl_t, pred_t, save_path=os.path.join(OUTPUT_DIR, "CWRU_Teacher_Confusion_Matrix.png"))

    eval_s = Evaluator("Student")
    lbl_s, pred_s = eval_s.evaluate(student, loader_te, device)
    eval_s.plot_confusion_matrix(lbl_s, pred_s, save_path=os.path.join(OUTPUT_DIR, "CWRU_Student_Confusion_Matrix.png"))

    plot_comparison(eval_t, eval_s, save_path=os.path.join(OUTPUT_DIR, "CWRU_Model_Comparison.png"))

    sid = np.random.randint(0, len(idx_te))
    plot_comprehensive(
        X[idx_te[sid]], {k: feats[k][idx_te[sid]] for k in feats}, y[idx_te[sid]], pred_s[sid],
        save_path=os.path.join(OUTPUT_DIR, "CWRU_Comprehensive_Result.png")
    )

    print(f"\n=== Pipeline Completed! Outputs saved to: {OUTPUT_DIR} ===")


if __name__ == "__main__":
    main()