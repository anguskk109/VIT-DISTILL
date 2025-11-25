import torch
import timm
from torchvision.transforms import Resize
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"
K = 5

# Load teacher
teacher = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=10)
state_dict = torch.load("cache/teacher_vit_small_cifar10.pth", map_location="cpu", weights_only=True)
teacher.load_state_dict(state_dict)
teacher.to(device).eval()

# ImageNet stats for timm
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
CIFAR_STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)

resizer = Resize(224)

def convert_to_imagenet_norm(cifar_tensor):
    # Denormalize from CIFAR
    x = cifar_tensor * CIFAR_STD + CIFAR_MEAN
    # Normalize to ImageNet
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x

Path("cache").mkdir(exist_ok=True)

for k in range(K):
    print(f"Computing logits for view {k+1}/{K}...")
    aug_data = torch.load(f"cache/cifar10_train_augmented_k{k}.pth", weights_only=True)  # (50000, 3, 32, 32)
    x = convert_to_imagenet_norm(aug_data)

    # Process in batches
    resized_batches = []
    batch_size = 1000
    for i in range(0, len(x), batch_size):
        batch = x[i:i+batch_size]
        resized_batch = torch.stack([resizer(img) for img in batch])  # (B, 3, 224, 224)
        resized_batches.append(resized_batch)

    all_logits = []
    with torch.no_grad():
        for resized_batch in resized_batches:
            resized_batch = resized_batch.to(device)
            logits = teacher(resized_batch)
            all_logits.append(logits.cpu())

    logits = torch.cat(all_logits, dim=0)  # (50000, 10)
    torch.save(logits, f"cache/teacher_logits_k{k}.pth")
    print(f"  Saved logits for view {k}")

print("Multi-view teacher logits precomputed.")