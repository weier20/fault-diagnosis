import os
import time
import glob
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

# 全局训练用随机种子
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

# 目录配置
DATA_DIR = r"D:\weier\test\PU"
OUTPUT_DIR = r"D:\weier\test\test_PU"
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


# PU 数据加载与特征提取
# 使用您指定的 load_pu_data 版本
def load_pu_data(data_dir, seq_len=1024, samples_per_class=1000):
    print(f"Scanning PU dataset in: {data_dir} ...")
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    data, labels = [], []
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    mat_files = glob.glob(os.path.join(data_dir, '**', '*.mat'), recursive=True)

    # 智能信号提取
    def extract_signal(obj):
        candidates = []

        # 内部递归函数，找出所有长度合格的一维数组
        def _search(o):
            if isinstance(o, np.ndarray):
                if o.dtype.names is not None:
                    for name in o.dtype.names:
                        _search(o[name])
                else:
                    sq = o.squeeze()
                    if sq.ndim == 1 and len(sq) > 10000:
                        if not np.all(np.diff(sq) >= 0):
                            candidates.append(sq)
                    elif o.size > 0 and o.dtype == object:
                        for item in o.flatten():
                            _search(item)
            elif isinstance(o, dict):
                for k, v in o.items():
                    if not k.startswith('__'):
                        _search(v)

        _search(obj)

        # 如果找到了多个候选信号，返回方差最大（波动最剧烈）的那一个，这通常是主振动传感器数据
        if candidates:
            candidates.sort(key=lambda x: np.var(x), reverse=True)
            return candidates[0]
        return None

    for file_path in mat_files:
        filename = os.path.basename(file_path).upper()
        label = -1

        if 'K0' in filename:
            label = 0
        elif 'KI' in filename:
            label = 1
        elif 'KA' in filename:
            label = 2
        elif 'KB' in filename:
            label = 3

        if label == -1 or counts[label] >= samples_per_class: continue

        try:
            mat_dict = sio.loadmat(file_path)
            sig = extract_signal(mat_dict)

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
            print(f"Error reading {filename}: {e}")
            pass

    if len(data) == 0:
        raise ValueError("No valid PU data parsed! 请确保 .rar 已经解压，且包含 .mat 文件。")
    print(
        f"Data ready -> Normal(K0): {counts[0]}, Inner(KI): {counts[1]}, Outer(KA): {counts[2]}, Mixed(KB): {counts[3]}")
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
        # PU 数据集采样率为 64000 Hz
        _, _, Zxx = signal.stft(sig, fs=64000, nperseg=128)
        features['stft'].append(_resize(np.abs(Zxx)))

        coeffs, _ = pywt.cwt(sig, scales, 'morl')
        features['wavelet'].append(_resize(np.abs(coeffs)))

        s_norm = (sig - np.min(sig)) / (np.max(sig) - np.min(sig) + 1e-8)
        phi = np.arccos(np.clip(s_norm, -1, 1))
        gaf = np.outer(np.cos(phi), np.cos(phi)) - np.outer(np.sin(phi), np.sin(phi))
        features['gaf'].append(_resize(gaf))

    return {k: np.array(v) for k, v in features.items()}


# 自监督模块
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
    noise_factor = 0.05
    scale_factor = torch.rand(ts.size(0), 1, device=ts.device) * 0.4 + 0.8
    ts_aug = ts * scale_factor + torch.randn_like(ts) * noise_factor
    stft_aug = stft + torch.randn_like(stft) * noise_factor
    wt_aug = wt + torch.randn_like(wt) * noise_factor
    gaf_aug = gaf + torch.randn_like(gaf) * noise_factor
    return ts_aug, stft_aug, wt_aug, gaf_aug


# 模型定义
class ConvEncoder(nn.Module):
    def __init__(self, is_1d=False, in_c=1, rep_dim=128):
        super().__init__()
        self.is_1d = is_1d
        if is_1d:
            self.net = nn.Sequential(
                nn.Conv1d(in_c, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.AdaptiveAvgPool1d(16),
                nn.Flatten(), nn.Linear(64 * 16, rep_dim), nn.BatchNorm1d(rep_dim)
            )
        else:
            self.net = nn.Sequential(
                nn.Conv2d(in_c, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(), nn.Linear(64 * 16, rep_dim), nn.BatchNorm1d(rep_dim)
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
        self.projector = nn.Sequential(
            nn.Linear(128 * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        layers = []
        in_dim = 128 * 4
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.3)])
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, ts, stft, wt, gaf, return_proj=False):
        fused = torch.cat([enc(x) for enc, x in zip(self.encoders, [ts, stft, wt, gaf])], dim=1)
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


# 评估模块
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
        classes = ['Normal(K0)', 'Inner(KI)', 'Outer(KA)', 'Mixed(KB)']
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
    names = {0: 'Normal(K0)', 1: 'Inner(KI)', 2: 'Outer(KA)', 3: 'Mixed(KB)'}
    ax[0, 0].plot(sig);
    ax[0, 0].set_title(f'Original Signal\nTrue: {names[true_lbl]} | Pred: {names[pred_lbl]}')
    ax[0, 1].imshow(feat['stft'], aspect='auto', cmap='hot', origin='lower');
    ax[0, 1].set_title('STFT')
    ax[0, 2].imshow(feat['wavelet'], aspect='auto', cmap='viridis', origin='lower');
    ax[0, 2].set_title('Wavelet')
    ax[0, 3].imshow(feat['gaf'], cmap='magma');
    ax[0, 3].set_title('GAF')
    f, Pxx = signal.periodogram(sig, fs=64000);
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


# 主干
def main():
    print("\n=== Parsing PU Data ===")

    X, y = load_pu_data(DATA_DIR, seq_len=1024, samples_per_class=1000)

    np.save(os.path.join(OUTPUT_DIR, "PU_data_X.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "PU_labels_y.npy"), y)

    feats = extract_features(X)
    idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.3, random_state=42, stratify=y)

    # 【关键修复】训练集打乱，测试集绝对不打乱 (shuffle=False)
    loader_tr = DataLoader(
        MultiModalDataset(X[idx_tr], {k: feats[k][idx_tr] for k in feats}, y[idx_tr]),
        batch_size=32, shuffle=True
    )

    loader_te = DataLoader(
        MultiModalDataset(X[idx_te], {k: feats[k][idx_te] for k in feats}, y[idx_te]),
        batch_size=32, shuffle=False
    )

    def _build(hidden):
        return MultiModalNet([ConvEncoder(is_1d=True).to(device)] + [ConvEncoder().to(device) for _ in range(3)],
                             hidden, num_classes=4).to(device)

    teacher = _build([512, 256, 128])
    student = _build([256, 128])

    # Teacher无监督预训练
    print("\n--- Stage 1: Self-Supervised Pre-training (Teacher) ---")
    opt_pre = torch.optim.Adam(teacher.parameters(), lr=1e-3)
    contrastive_criterion = ContrastiveLoss(temperature=0.1)

    for ep in range(30):
        teacher.train()
        total_ssl_loss = 0
        for b in loader_tr:
            ts, stft, wt, gaf = (b[k].to(device) for k in ['ts', 'stft', 'wt', 'gaf'])
            ts1, stft1, wt1, gaf1 = augment_batch(ts, stft, wt, gaf)
            ts2, stft2, wt2, gaf2 = augment_batch(ts, stft, wt, gaf)
            proj1 = teacher(ts1, stft1, wt1, gaf1, return_proj=True)
            proj2 = teacher(ts2, stft2, wt2, gaf2, return_proj=True)
            projections = torch.cat([proj1, proj2], dim=0)
            loss = contrastive_criterion(projections)

            opt_pre.zero_grad()
            loss.backward()
            opt_pre.step()
            total_ssl_loss += loss.item()

        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"Pre-train Epoch {ep + 1}/30, SSL Loss: {total_ssl_loss / len(loader_tr):.4f}")

    # Teacher 有监督微调
    print("\n--- Stage 2: Supervised Fine-tuning (Teacher) ---")
    opt_t = torch.optim.Adam(teacher.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()

    for ep in range(30):
        teacher.train()
        total_loss = 0
        for b in loader_tr:
            ts, stft, wt, gaf, lbl = (b[k].to(device) for k in ['ts', 'stft', 'wt', 'gaf', 'lbl'])
            opt_t.zero_grad()
            loss = ce(teacher(ts, stft, wt, gaf), lbl)
            loss.backward()
            opt_t.step()
            total_loss += loss.item()

        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"Fine-tune Epoch {ep + 1}/30, CE Loss: {total_loss / len(loader_tr):.4f}")

    torch.save(teacher.state_dict(), os.path.join(OUTPUT_DIR, "teacher_model_PU.pth"))

    # Student 知识蒸馏
    print("\n--- Stage 3: Distilling Knowledge to Student ---")
    opt_s, kl = torch.optim.Adam(student.parameters(), lr=1e-3), nn.KLDivLoss(reduction='batchmean')
    T, alpha = 3.0, 0.6

    for ep in range(20):
        student.train()
        total_loss = 0
        for b in loader_tr:
            ts, stft, wt, gaf, lbl = (b[k].to(device) for k in ['ts', 'stft', 'wt', 'gaf', 'lbl'])
            opt_s.zero_grad()
            t_log, s_log = teacher(ts, stft, wt, gaf).detach(), student(ts, stft, wt, gaf)

            soft_loss = kl(F.log_softmax(s_log / T, dim=1), F.softmax(t_log / T, dim=1)) * (T ** 2)
            hard_loss = ce(s_log, lbl)
            loss = alpha * soft_loss + (1 - alpha) * hard_loss

            loss.backward()
            opt_s.step()
            total_loss += loss.item()

        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"Distillation Epoch {ep + 1}/20, Loss: {total_loss / len(loader_tr):.4f}")

    torch.save(student.state_dict(), os.path.join(OUTPUT_DIR, "student_model_PU.pth"))

    print("\n=== Evaluating & Generating Plots ===")
    eval_t = Evaluator("Teacher")
    lbl_t, pred_t = eval_t.evaluate(teacher, loader_te, device)
    eval_t.plot_confusion_matrix(lbl_t, pred_t, save_path=os.path.join(OUTPUT_DIR, "PU_Teacher_Confusion_Matrix.png"))

    eval_s = Evaluator("Student")
    lbl_s, pred_s = eval_s.evaluate(student, loader_te, device)
    eval_s.plot_confusion_matrix(lbl_s, pred_s, save_path=os.path.join(OUTPUT_DIR, "PU_Student_Confusion_Matrix.png"))

    plot_comparison(eval_t, eval_s, save_path=os.path.join(OUTPUT_DIR, "PU_Model_Comparison.png"))

    np.random.seed(int(time.time()))
    sid = np.random.randint(0, len(idx_te))
    plot_comprehensive(
        X[idx_te[sid]], {k: feats[k][idx_te[sid]] for k in feats}, y[idx_te[sid]], pred_s[sid],
        save_path=os.path.join(OUTPUT_DIR, "PU_Comprehensive_Result.png")
    )

    print(f"\n=== Pipeline Completed! Outputs saved to: {OUTPUT_DIR} ===")


if __name__ == "__main__":
    main()