import torch
import time
import json
import argparse
from pathlib import Path

from student_vit import StudentViT
from utils import get_dataloaders, train_epoch_multiview, evaluate, set_seed, EarlyStopping

def main():

    # ----------
    # configurable hyperparameters
    # ----------
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--min_epochs", type=int, default=400)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--T", type=float, default=4.0)           # distillation temp
    parser.add_argument("--alpha", type=float, default=0.8)       # distill vs CE weight
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to full checkpoint to resume training")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Print hyperparameters
    print("\n" + "="*50)
    print("Starting Knowledge Distillation Training")
    print("="*50)
    for arg, value in vars(args).items():
        print(f"  {arg:<20}: {value}")
    print(f"  {'device':<20}: {device}")
    print("="*50 + "\n")

    set_seed(args.seed)

    # ----------------------------
    # train/val split (90/10)
    # ----------------------------
    # Load labels (same for all views)
    train_targets = torch.load("cache/cifar10_train_labels.pth", weights_only=True)  # (50000,)

    # Train/val split
    total = 50000
    train_size = int(0.9 * total)
    generator = torch.Generator().manual_seed(args.seed)
    train_indices, val_indices = torch.utils.data.random_split(
        range(total), [train_size, total - train_size], generator=generator
    )
    train_indices = torch.tensor(train_indices.indices)
    val_indices = torch.tensor(val_indices.indices)

    # Load val data (use view 0 for validation)
    val_inputs = torch.load("cache/cifar10_train_augmented_k0.pth", weights_only=True)[val_indices]
    val_targets = train_targets[val_indices]
    val_dataset = torch.utils.data.TensorDataset(val_inputs, val_targets)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    _, test_loader = get_dataloaders(batch_size=args.batch_size, num_workers=6)

    print("\nTraining Student with Knowledge Distillation...")
    # Student config
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
    # Training Loop
    # ----------------------------
    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True)
    checkpoint_path = results_dir / "student_distill_checkpoint_latest.pth"
    best_model_path = results_dir / "student_distill_multiview.pth"

    K = 5
    start_epoch = 0
    best_val_acc = 0.0
    early_stopper = EarlyStopping(patience=args.patience, delta=1e-4, mode="max", verbose=True)
    train_losses, val_accs = [], []
    

    start_time = time.time()

    if args.resume_from:
        print(f"Loading checkpoint: {args.resume_from}")
        ckpt = torch.load(args.resume_from, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_acc = ckpt.get('best_val_acc', 0.0)
        train_losses = ckpt.get('train_losses', [])
        val_accs = ckpt.get('val_accs', [])
        print(f"   Resuming from epoch {start_epoch}, best val acc: {best_val_acc:.4f}")

    # Override early stopper's best score if resuming
    early_stopper.best_score = best_val_acc if args.resume_from else None

    for epoch in range(start_epoch, args.epochs):
        # Randomly pick a view
        view_id = torch.randint(0, K, (1,)).item()
        
        # Load FULL augmented data and logits for this view
        full_inputs = torch.load(f"cache/cifar10_train_augmented_k{view_id}.pth", weights_only=True)      # (50000, ...)
        full_logits = torch.load(f"cache/teacher_logits_k{view_id}.pth", weights_only=True)              # (50000, 10)
        
        # Extract SUBSET using global train_indices
        train_inputs = full_inputs[train_indices]        # (45000, ...)
        train_t_logits = full_logits[train_indices]      # (45000, 10)
        train_targets_split = train_targets[train_indices]  # (45000,)

        # create shuffled LOCAL indices (0 to 44999)
        local_indices = torch.randperm(len(train_inputs)).tolist()
        shuffled_batches = [
            local_indices[i:i + args.batch_size]
            for i in range(0, len(local_indices), args.batch_size)
        ]

        avg_loss = train_epoch_multiview(
            model, shuffled_batches, train_inputs, train_targets_split,
            train_t_logits, optimizer, device, scaler, args.T, args.alpha
        )
        scheduler.step()

        val_acc = evaluate(model, val_loader, device)
        train_losses.append(avg_loss)
        val_accs.append(val_acc)

        if (epoch + 1) % 10 == 0 or epoch >= args.min_epochs:
            print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, ValAcc={val_acc:.4f}")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_losses': train_losses,
            'val_accs': val_accs,
            'best_val_acc': max(val_accs) if val_accs else 0.0,
        }, checkpoint_path)

        if epoch >= args.min_epochs:
            early_stopper(val_acc, model=model, model_path=best_model_path)
            if early_stopper.early_stop:
                print("Early stopping triggered.")
                break

    # Final test
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    final_test_acc = evaluate(model, test_loader, device)

    # Save results
    elapsed = time.time() - start_time
    results = {
        "experiment": "distill_multiview",
        "config": {**student_config, **vars(args)},
        "epochs_run": len(val_accs),
        "best_val_acc": max(val_accs),
        "final_test_acc": final_test_acc,
        "training_time_sec": elapsed,
    }
    with open(results_dir / "distill_multiview.json", "w") as f:
        json.dump(results, f, indent=2)
    torch.save(model.state_dict(), results_dir / "student_final.pth")

    print(f"\nTraining completed in {elapsed:.2f}s")
    print(f"Best val accuracy: {max(val_accs):.4f}")
    print(f"Final TEST accuracy: {final_test_acc:.4f}")
    print(f"Results saved to {results_dir}")

if __name__ == "__main__":
    main()