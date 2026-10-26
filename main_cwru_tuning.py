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

# Directory Configuration
DATA_DIR = r"./data/cwru_data"
OUTPUT_DIR = r"./results"
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


#  Data Loading & Cross-Condition Split
def load_cwru_data_with_val(data_dir, seq_len=1024):
    print(f"Scanning CWRU dataset in: {data_dir} ...")
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    mat_files = glob.glob(os.path.join(data_dir, '**', '*.mat'), recursive=True)

    # Disjoint file split across operating conditions (0HP/1HP: Train, 2HP: Val, 3HP: Test)
    train_list = [97, 98, 105, 106, 118, 119, 130, 131, 144, 145, 156, 158]
    val_list = [99, 107, 120, 132, 146, 159]
    test_list = [100, 108, 121, 133, 147, 160]

    def get_label_by_num(num):
        if num in [97, 98, 99, 100]: return 0
        if num in [105, 106, 107, 108, 169, 170, 171, 172]: return 1
        if num in [130, 131, 132, 133, 144, 145, 146, 147, 156, 158, 159, 160]: return 2
        if num in [118, 119, 120, 121, 185, 186, 187, 188]: return 3
        return -1

    X_tr, y_tr = [], []
    X_val, y_val = [], []
    X_te, y_te = [], []

    counts_tr = {0: 0, 1: 0, 2: 0, 3: 0}
    counts_val = {0: 0, 1: 0, 2: 0, 3: 0}
    counts_te = {0: 0, 1: 0, 2: 0, 3: 0}

    for file_path in mat_files:
        filename = os.path.basename(file_path)
        match = re.search(r'(\d+)', filename)
        if not match: continue
        num = int(match.group(1))

        lbl = get_label_by_num(num)
        if lbl == -1: continue

        partition = None
        if num in train_list:
            partition = 'train'
        elif num in val_list:
            partition = 'val'
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
                    start = i * seq_len
                    window_sig = sig[start: start + seq_len]
                    window_sig = (window_sig - np.mean(window_sig)) / (np.std(window_sig) + 1e-8)

                    if partition == 'train':
                        X_tr.append(window_sig)
                        y_tr.append(lbl)
                        counts_tr[lbl] += 1
                    elif partition == 'val':
                        X_val.append(window_sig)
                        y_val.append(lbl)
                        counts_val[lbl] += 1
                    elif partition == 'test':
                        X_te.append(window_sig)
                        y_te.append(lbl)
                        counts_te[lbl] += 1
        except Exception:
            pass

    print("\n=== Data Split Manifest ===")
    print(f"Train Partition (0HP & 1HP) -> Total Windows: {sum(counts_tr.values())}")
    print(f"Validation Partition (2HP) -> Total Windows: {sum(counts_val.values())}")
    print(f"Test Partition (3HP) -> Total Windows: {sum(counts_te.values())}")
    print("===========================\n")

    X = np.concatenate([np.array(X_tr), np.array(X_val), np.array(X_te)], axis=0)
    y = np.concatenate([np.array(y_tr), np.array(y_val), np.array(y_te)], axis=0)

    idx_tr = np.arange(len(X_tr))
    idx_val = np.arange(len(X_tr), len(X_tr) + len(X_val))
    idx_te = np.arange(len(X_tr) + len(X_val), len(X))

    return X, y, idx_tr, idx_val, idx_te


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
        features['gaf'].append(_resize(np.outer(np.cos(phi), np.cos(phi)) - np.outer(np.sin(phi), np.sin(phi))))
    return {k: np.array(v) for k, v in features.items()}


# Model Components
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
    def __init__(self, encoders, hidden_dims, num_classes=4):
        super().__init__()
        self.encoders = nn.ModuleList(encoders)
        self.projector = nn.Sequential(nn.Linear(128 * 4, 256), nn.ReLU(), nn.Linear(256, 128))
        layers = []
        in_dim = 128 * 4
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(0.3)])
            in_dim = h
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
        self.metrics = {'accuracy': acc}
        return labels, preds

    def plot_confusion_matrix(self, labels, preds, save_path=None):
        cm = confusion_matrix(labels, preds)
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        classes = ['Normal', 'IR Fault', 'OR Fault', 'Ball Fault']
        ax.set(xticks=np.arange(4), yticks=np.arange(4), xticklabels=classes, yticklabels=classes,
               title=f'{self.name} - Confusion Matrix (Test)', ylabel='True', xlabel='Predicted')
        for i in range(4):
            for j in range(4):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2. else "black")
        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


def plot_hyperparameter_tuning(results, save_path):
    labels = [f"T={r['T']}, α={r['alpha']}" for r in results]
    accs = [r['val_acc'] * 100 for r in results]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, accs, color='skyblue', edgecolor='black')

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval - 1.5, f"{yval:.2f}%", ha='center', va='bottom', color='white',
                 fontweight='bold')

    plt.ylim(min(accs) - 2, 100)
    plt.ylabel('Validation Accuracy (%)')
    plt.title('Hyperparameter Tuning on Validation Set (2 HP)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    print("\n=== Parsing CWRU Data with Strict Mutual Exclusion ===")
    X, y, idx_tr, idx_val, idx_te = load_cwru_data_with_val(DATA_DIR, seq_len=1024)

    feats = extract_features(X)

    def _loader(idx, is_train=True):
        return DataLoader(MultiModalDataset(X[idx], {k: feats[k][idx] for k in feats}, y[idx]), batch_size=32,
                          shuffle=is_train)

    loader_tr = _loader(idx_tr, is_train=True)
    loader_val = _loader(idx_val, is_train=False)
    loader_te = _loader(idx_te, is_train=False)

    def _build(hidden):
        return MultiModalNet([ConvEncoder(is_1d=True).to(device)] + [ConvEncoder().to(device) for _ in range(3)],
                             hidden).to(device)

    teacher = _build([512, 256, 128])

    print("\n--- Stage 1: Self-Supervised Pre-training (Teacher) ---")
    opt_pre = torch.optim.Adam(teacher.parameters(), lr=1e-3)
    contrastive_criterion = ContrastiveLoss(temperature=0.1)

    for ep in range(5):
        teacher.train()
        for b in loader_tr:
            ts, stft, wt, gaf = (b[k].to(device) for k in ['ts', 'stft', 'wt', 'gaf'])
            ts1, stft1, wt1, gaf1 = augment_batch(ts, stft, wt, gaf)
            ts2, stft2, wt2, gaf2 = augment_batch(ts, stft, wt, gaf)
            loss = contrastive_criterion(
                torch.cat([teacher(ts1, stft1, wt1, gaf1, True), teacher(ts2, stft2, wt2, gaf2, True)], dim=0))
            opt_pre.zero_grad()
            loss.backward()
            opt_pre.step()

    print("\n--- Stage 2: Supervised Fine-tuning (Teacher) ---")
    opt_t = torch.optim.Adam(teacher.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()

    best_t_val_acc = 0.0
    for ep in range(5):
        teacher.train()
        for b in loader_tr:
            ts, stft, wt, gaf, lbl = (b[k].to(device) for k in ['ts', 'stft', 'wt', 'gaf', 'lbl'])
            opt_t.zero_grad()
            ce(teacher(ts, stft, wt, gaf), lbl).backward()
            opt_t.step()

        eval_t_val = Evaluator("Teacher_Val")
        _, _ = eval_t_val.evaluate(teacher, loader_val, device)
        val_acc = eval_t_val.metrics['accuracy']
        if val_acc > best_t_val_acc:
            best_t_val_acc = val_acc
            torch.save(teacher.state_dict(), os.path.join(OUTPUT_DIR, "teacher_model_best.pth"))

    teacher.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "teacher_model_best.pth"), weights_only=True))
    print(f"Teacher Best Validation Acc: {best_t_val_acc:.4f}")

    # Grid Search for Hyperparameter Selection

    hyperparams_grid = [
        (3.0, 0.0),  
        (3.0, 1.0),  
        (1.0, 0.6), 
        (3.0, 0.6), 
        (5.0, 0.6),  
        (10.0, 0.6), 
        (3.0, 0.2), 
        (3.0, 0.8)  
    ]

    tuning_results = []
    best_overall_val_acc = 0.0
    best_params = None

    for T, alpha in hyperparams_grid:
        print(f"\nEvaluating Hyperparameters -> T: {T}, alpha: {alpha}")

        student = _build([256, 128])
        opt_s = torch.optim.Adam(student.parameters(), lr=1e-3)
        kl = nn.KLDivLoss(reduction='batchmean')

        best_s_val_acc_current = 0.0

        for ep in range(10):
            student.train()
            for b in loader_tr:
                ts, stft, wt, gaf, lbl = (b[k].to(device) for k in ['ts', 'stft', 'wt', 'gaf', 'lbl'])
                opt_s.zero_grad()
                t_log, s_log = teacher(ts, stft, wt, gaf).detach(), student(ts, stft, wt, gaf)
                loss = alpha * (kl(F.log_softmax(s_log / T, dim=1), F.softmax(t_log / T, dim=1)) * (T ** 2)) + (
                            1 - alpha) * ce(s_log, lbl)
                loss.backward()
                opt_s.step()

            eval_s_val = Evaluator("Student_Val")
            _, _ = eval_s_val.evaluate(student, loader_val, device)
            val_acc_s = eval_s_val.metrics['accuracy']

            if val_acc_s > best_s_val_acc_current:
                best_s_val_acc_current = val_acc_s
                if val_acc_s > best_overall_val_acc:
                    best_overall_val_acc = val_acc_s
                    best_params = (T, alpha)
                    torch.save(student.state_dict(), os.path.join(OUTPUT_DIR, "student_model_best_tuned.pth"))

        print(f"Result -> Best Validation Acc for (T={T}, α={alpha}): {best_s_val_acc_current:.4f}")
        tuning_results.append({'T': T, 'alpha': alpha, 'val_acc': best_s_val_acc_current})

    print(f"\nGrid Search Completed! Selected Optimal Parameters: T={best_params[0]}, alpha={best_params[1]}")
    print(f"Highest Validation Accuracy: {best_overall_val_acc:.4f}")

    plot_hyperparameter_tuning(tuning_results, os.path.join(OUTPUT_DIR, "CWRU_Hyperparameter_Tuning.png"))

    # Final Evaluation on Unseen Test Set (3 HP)
    print("\n=== Evaluating on Unseen TEST Data (3 HP) ===")
    student = _build([256, 128])
    student.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "student_model_best_tuned.pth"), weights_only=True))

    eval_t = Evaluator("Teacher")
    lbl_t, pred_t = eval_t.evaluate(teacher, loader_te, device)
    eval_t.plot_confusion_matrix(lbl_t, pred_t,
                                 save_path=os.path.join(OUTPUT_DIR, "CWRU_Teacher_Confusion_Matrix_Test.png"))
    print(f"Teacher Final Test Acc: {eval_t.metrics['accuracy']:.4f}")

    eval_s = Evaluator("Student_Optimal_HP")
    lbl_s, pred_s = eval_s.evaluate(student, loader_te, device)
    eval_s.plot_confusion_matrix(lbl_s, pred_s,
                                 save_path=os.path.join(OUTPUT_DIR, "CWRU_Student_Confusion_Matrix_Test.png"))
    print(f"Student Final Test Acc (using T={best_params[0]}, α={best_params[1]}): {eval_s.metrics['accuracy']:.4f}")

    print(f"\n=== Pipeline Completed! Outputs saved to: {OUTPUT_DIR} ===")


if __name__ == "__main__":
    main()
