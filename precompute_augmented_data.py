import torch
from torchvision import datasets, transforms
from pathlib import Path

K = 5  # number of augmented views
mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

raw_train = datasets.CIFAR10(root='./data', train=True, download=True)
labels = torch.tensor([label for _, label in raw_train])

Path("cache").mkdir(exist_ok=True)
torch.save(labels, "cache/cifar10_train_labels.pth")

for k in range(K):
    print(f"Generating view {k+1}/{K}...")
    torch.manual_seed(666 + k)  # different seed per view
    aug_imgs = []
    for img, _ in raw_train:
        aug_imgs.append(train_transform(img))
    aug_tensor = torch.stack(aug_imgs)  # (50000, 3, 32, 32)
    torch.save(aug_tensor, f"cache/cifar10_train_augmented_k{k}.pth")
    print(f"  Saved view {k}")

print("Multi-view data precomputed.")