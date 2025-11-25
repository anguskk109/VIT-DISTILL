import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import os
import json
from tqdm import tqdm
from torchvision.transforms import Resize

from utils import evaluate_with_resize, get_dataloaders

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load pretrained ViT-Base
    model = timm.create_model('vit_small_patch16_224', pretrained=True)
    model.head = nn.Linear(model.head.in_features, 10)
    model = model.to(device)

    # Freeze backbone, unfreeze head
    for param in model.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = True

    train_loader, test_loader = get_dataloaders(batch_size=512, num_workers=8)
    resizer = Resize(224)

    optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-3, weight_decay=1e-4)
    model.train()

    print("Fine-tuning heads on CIFAR-10...")
    for epoch in range(2):
        total_loss = 0
        for imgs, labels in tqdm(train_loader, desc="Fine Tuning"):
            # imgs: (B, C, 32, 32)
            imgs_224 = torch.stack([resizer(img) for img in imgs])  # (B, C, 224, 224)
            imgs_224 = imgs_224.to(device)
            labels = labels.to(device)
            logits = model(imgs_224)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Avg Loss: {total_loss/len(train_loader):.4f}")

    print("Evaluating teacher model on CIFAR-10...")
    teacher_model_acc = evaluate_with_resize(model, test_loader, device)
    print(f"Teacher_model Accuracy: {teacher_model_acc:.4f}")
    
    os.makedirs("results", exist_ok=True)
    os.makedirs("cache", exist_ok=True)
    results = {
        "experiment": "finetuned_teacher",
        "final_acc": teacher_model_acc,
    }

    with open("results/finetuned_teacher.json", "w") as f:
        json.dump(results, f, indent=2)

    torch.save(model.state_dict(), "cache/teacher_vit_small_cifar10.pth")
    print("Teacher saved.")

if __name__ == "__main__":
    main()