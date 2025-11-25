import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torchvision.transforms import Resize
from torch.utils.data import DataLoader

from tqdm import tqdm

def get_dataloaders(batch_size=128, num_workers=4):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    return train_loader, test_loader

def distillation_loss(student_logits, teacher_logits, labels, T=4.0, alpha=0.7):
    kd_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction='batchmean'
    ) * (T*T)
    ce_loss = F.cross_entropy(student_logits, labels, label_smoothing=0.1)
    return alpha * kd_loss + (1.0 - alpha) * ce_loss

def train_epoch_multiview(
    model, shuffled_batches, inputs, targets, t_logits, optimizer, device, scaler, T, alpha
):
    model.train()
    total_loss = 0.0
    for batch_indices in tqdm(shuffled_batches, desc="Training"):  # each is a list of indices, size = actual batch_size
        x = inputs[torch.tensor(batch_indices)].to(device, non_blocking=True)
        y = targets[torch.tensor(batch_indices)].to(device, non_blocking=True)
        t = t_logits[torch.tensor(batch_indices)].to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type, enabled=scaler is not None):
            s = model(x)
            loss = distillation_loss(s, t, y, T=T, alpha=alpha)

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(shuffled_batches)

def train_epoch_offline(
    model,
    shuffled_batches,
    train_inputs,
    train_targets,
    train_teacher_logits,
    optimizer,
    device,
    scaler=None,
    T=4.0,
    alpha=0.7
):
    model.train()
    total_loss = 0.0
    for batch_local_indices in tqdm(shuffled_batches, desc="Training"):
        batch_local_indices = torch.tensor(batch_local_indices)

        inputs = train_inputs[batch_local_indices].to(device, non_blocking=True)
        labels = train_targets[batch_local_indices].to(device, non_blocking=True)
        t_logits = train_teacher_logits[batch_local_indices].to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type, enabled=(scaler is not None)):
            s_logits = model(inputs)
            loss = distillation_loss(s_logits, t_logits, labels, T=T, alpha=alpha)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(shuffled_batches)

def train_epoch_scratch(student, loader, optimizer, device, scaler=None):
    student.train()
    total_loss = 0

    for imgs, labels in tqdm(loader, desc="Training"):
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type, enabled=(scaler is not None)):
            logits = student(imgs)
            loss = F.cross_entropy(logits, labels, label_smoothing=0.1)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:            
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(imgs)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return correct / total

@torch.no_grad()
def evaluate_with_resize(model, loader, device):
    """Evaluate model that expects larger input (e.g., ViT-Base) on CIFAR-10."""
    model.eval()
    correct = 0
    total = 0
    resizer = Resize(224)
    for imgs, labels in tqdm(loader, desc="Eval Teacher", leave=False):
        imgs_resized = torch.stack([resizer(img) for img in imgs]).to(device)        
        labels = labels.to(device)
        logits = model(imgs_resized)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return correct / total


def set_seed(seed=666):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class EarlyStopping:
    def __init__(self, patience=10, delta=0, mode="min", verbose=False):
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, metric, model=None, model_path=None):
        score = metric if self.mode == "max" else -metric
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(metric, model, model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(metric, model, model_path)
            self.counter = 0
        

    def save_checkpoint(self, metric, model, model_path):
        if model is not None and model_path is not None:
            torch.save(model.state_dict(), model_path)
            if self.verbose:
                print(f"New best model (metric={metric:.5f}) saved to {model_path}")
        