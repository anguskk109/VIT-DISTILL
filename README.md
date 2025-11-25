# Knowledge Distillation on CIFAR-10 with Precomputed Multi-View Teacher Logits

This repository implements knowledge distillation (KD) from a ViT teacher to a smaller student model on CIFAR-10.  
Instead of computing teacher logits on-the-fly during training (which is slow), we **precompute teacher logits for multiple augmented views** of each image — then use them during student training to simulate multi-view distillation.

> ✅ Why? To reduce overfitting from single-view KD while avoiding expensive real-time teacher inference.

---

## 📁 Project Structure
├── cache/ # Precomputed teacher logits & augmented data
│ ├── cifar10_train_augmented_k*.pth # Augmented images (k=0~4)
│ ├── teacher_logits_k*.pth # Teacher logits for each view
│ └── teacher_vit_small_cifar10.pth # Finetuned ViT teacher model
├── data/ # CIFAR-10 raw data (downloaded separately)
├── results/ # Training metrics & final checkpoints
│ ├── distill_multiview.json # Final accuracy/metrics
│ ├── finetuned_teacher.json # Teacher fine-tuning metrics
│ ├── student_distill_multiview.pth # Best distilled student model
│ ├── student_final.pth # Final student checkpoint
│ └── student_scratch.json # Student trained from scratch (baseline)
├── requirements.txt # Python dependencies
├── distill_train.py # Main script: trains student with precomputed logits
├── finetune_teacher.py # Fine-tunes ViT teacher on CIFAR-10
├── precompute_augmented_data.py # Generates k=5 augmented views per image
├── precompute_teacher_logits.py # Computes teacher logits for all augmented views
├── scratch_train.py # Trains student from scratch (no KD)
├── student_vit.py # Defines student model (ViT)
└── utils.py # Helper functions

## ⚙️ Setup & Usage
Precompute Teacher Logits (One-Time Setup)

First, train/fine-tune the teacher model:
python finetune_teacher.py

Then generate 5 augmented views per image:
python precompute_augmented_data.py

Finally, compute teacher logits for all views:
python precompute_teacher_logits.py
→ This will populate cache/ with .pth files.

Train with distillation using precomputed multi-view logits:
python distill_train.py

Or train from scratch (baseline):
python scratch_train.py

Results
Check results/distill_multiview.json and results/student_scratch.json for final accuracy.

## ✨Final Results
| Method             | StudentAcc(%) | TeacherAcc(%) |
|--------------------|---------------|---------------|
| Baseline (CE)      | 88.6          | –             |
| KD (Single View)   | 71.1          | 92.8          |
| KD (Multi-View)    | **87.1**      | 92.8          |

> 💡 Multi-view distillation brings student performance close to baseline — while still benefiting from teacher knowledge!

## 🧠 Key Design Choices
Precomputation: Teacher logits are computed once and reused → faster training.
Multi-view: We generate 5 different augmentations per image → student sees “different views” of same sample → reduces overfitting.
No online teacher: Avoids GPU memory pressure from running teacher during student training.
