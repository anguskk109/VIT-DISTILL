# Knowledge Distillation on CIFAR-10 with Precomputed Multi-View Teacher Logits

This repository implements knowledge distillation (KD) from a ViT teacher to a smaller student model on CIFAR-10.  
Instead of computing teacher logits on-the-fly during training (which is slow), we **precompute teacher logits for multiple augmented views** of each image — then use them during student training to simulate multi-view distillation.

> ✅ Why? To reduce overfitting from single-view KD while avoiding expensive real-time teacher inference.

---

## 📁 Project Structure
- **`cache/`** – Precomputed teacher logits and augmented data  *(not tracked in Git)*
  - `cifar10_train_augmented_k*.pth`: 5 augmented views per image (k=0–4)  
  - `teacher_logits_k*.pth`: Teacher logits for each view  
  - `teacher_vit_small_cifar10.pth`: Trained ViT teacher model

- **`data/`** – CIFAR-10 raw dataset *(not tracked in Git)*

- **`results/`** – Training outputs  
  - `distill_multiview.json`: Metrics for multi-view KD  
  - `student_scratch.json`: Baseline (no distillation)  
  - `*.pth`: Model checkpoints

- **Scripts**  
  - `finetune_teacher.py`: Fine-tune ViT on CIFAR-10  
  - `precompute_augmented_data.py`: Generate augmented views  
  - `precompute_teacher_logits.py`: Run teacher inference  
  - `distill_train.py`: Train student with precomputed logits  
  - `scratch_train.py`: Train student from scratch  
  - `student_vit.py`: Student model definition  
  - `utils.py`:  helpers

- **Config & Docs**  
  - `requirements.txt`: Python dependencies (pip)  
  - `environment.yml`: Conda environment (recommended)  
  - `README.md`: This file  
  - `.gitignore`: Ignored files

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
