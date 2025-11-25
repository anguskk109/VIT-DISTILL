import torch
import json
import time
import argparse
from pathlib import Path

from student_vit import StudentViT
from utils import EarlyStopping, evaluate, train_epoch_scratch, set_seed, get_dataloaders


def main():
    # ----------
    # Configurable hyperparameters
    # ----------
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--min_epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ----------------------------
    # train/val split (90/10)
    # ----------------------------
    from torchvision import datasets, transforms

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    # Raw dataset (PIL images)
    raw_train = datasets.CIFAR10(root='./data', train=True, download=True)
    test_loader = get_dataloaders(batch_size=args.batch_size, num_workers=8)[1]

    # Split indices
    total = len(raw_train)
    train_size = int(0.9 * total)
    val_size = total - train_size
    generator = torch.Generator().manual_seed(args.seed)
    train_indices, val_indices = torch.utils.data.random_split(
        range(total), [train_size, val_size], generator=generator
    )
    train_indices = train_indices.indices
    val_indices = val_indices.indices

    # Define transforms
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Build datasets
    train_dataset = torch.utils.data.Subset(
        datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform),
        train_indices
    )
    val_dataset = torch.utils.data.Subset(
        datasets.CIFAR10(root='./data', train=True, download=True, transform=base_transform),
        val_indices
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8,
        pin_memory=True, persistent_workers=True, prefetch_factor=2
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True
    )

    train_dataset_no_aug = torch.utils.data.Subset(
        datasets.CIFAR10(root='./data', train=True, download=True, transform=base_transform),
        train_indices
    )
    train_loader_no_aug = torch.utils.data.DataLoader(
        train_dataset_no_aug, batch_size=args.batch_size, shuffle=True, num_workers=8,
        pin_memory=True, persistent_workers=True, prefetch_factor=2
    )

    # ----------------------------
    # Model + Optimizer + Scheduler
    # ----------------------------
    student_config = dict(
        img_size=32,
        patch_size=4,
        in_chans=3,
        embed_dim=192,
        depth=6,
        num_heads=3,
        mlp_ratio=4.0,
        num_classes=10,
        drop_rate=0.2,
        attn_drop_rate=0.1
    )
    model = StudentViT(**student_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler()

    from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
    warmup_epochs = 10
    scheduler1 = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    scheduler2 = CosineAnnealingLR(optimizer, T_max=args.epochs - warmup_epochs)
    scheduler = SequentialLR(optimizer, [scheduler1, scheduler2], milestones=[warmup_epochs])

    # ----------------------------
    # Training Loop with Early Stopping
    # ----------------------------
    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True)
    best_model_path = results_dir / "student_scratch_best.pth"

    early_stopper = EarlyStopping(patience=args.patience, delta=1e-4, mode="max", verbose=True)
    train_losses, val_accs = [], []

    start_time = time.time()
    for epoch in range(args.epochs):
        loss = train_epoch_scratch(model, train_loader, optimizer, device, scaler=scaler)
        val_acc = evaluate(model, val_loader, device)
        scheduler.step()

        train_losses.append(loss)
        val_accs.append(val_acc)

        if (epoch+1) % 10 == 0:
            train_acc = evaluate(model, train_loader_no_aug, device)
            print(f"Epoch {epoch+1}: Loss={loss:.4f}, TrainAcc={train_acc:.4f}, ValAcc={val_acc:.4f}")
        
        if epoch < args.min_epochs:
            continue
        early_stopper(val_acc, model=model, model_path=best_model_path)
        if early_stopper.early_stop:
            print("Early stopping triggered.")
            break

    # ----------------------------
    # Final evaluation on TEST SET
    # ----------------------------
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    final_test_acc = evaluate(model, test_loader, device)

    # ----------------------------
    # Save results
    # ----------------------------
    elapsed = time.time() - start_time
    results = {
        "experiment": "student_scratch",
        "config": {**student_config, **vars(args)},
        "epochs_run": len(val_accs),
        "stopped_early": early_stopper.early_stop,
        "train_losses": train_losses,
        "val_accs": val_accs,
        "best_val_acc": max(val_accs),
        "final_test_acc": final_test_acc,
        "training_time_sec": elapsed,
    }

    with open(results_dir / "student_scratch.json", "w") as f:
        json.dump(results, f, indent=2)
    torch.save(model.state_dict(), results_dir / "student_scratch_final.pth")

    print(f"\nTraining completed in {elapsed:.2f}s")
    print(f"Best val accuracy: {max(val_accs):.4f}")
    print(f"Final TEST accuracy: {final_test_acc:.4f}")


if __name__ == "__main__":
    main()