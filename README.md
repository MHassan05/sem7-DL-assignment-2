# sem7-DL-assignment-2

Deep Learning assignment, Section B. A small Inception-style CNN is trained from scratch on 500 CIFAR-10 images and evaluated on a clean and a shifted test set.

## Setup
- Data: CIFAR-10 via `torchvision` (downloaded automatically).
- `train_small`: 500 training images (50 per class), split into 400 train and 100 validation.
- `clean_test`: the original 10,000 test images.
- `shifted_test`: the same images with random rotation (±25°), brightness and contrast changes, and a 16×16 black occlusion box (25% of the image).
- Test sets are used for reporting only. All hyperparameters were chosen on the validation split.

## Seeds
| Purpose | Seed |
|---|---|
| Data selection and split | `SPLIT_SEED = 42` |
| Shifted test set | `SHIFT_SEED = 123` |
| Training runs (mean ± std) | `RUN_SEEDS = [0, 1, 2]` |

## Model and training
- Inception-style CNN, 145,242 trainable parameters, trained from scratch.
- AdamW (lr 3e-3, weight decay 5e-4), cosine LR schedule, 60 epochs, batch size 64, label smoothing 0.1, Mixup (alpha 0.2), best-validation checkpoint.
- Augmentation (strength 0.25, applied on the fly to training data only): random crop, horizontal flip, rotation, color jitter, Gaussian noise, Random Erasing.

## Results (mean ± std over 3 seeds)
| Configuration | Val Acc | Clean Test | Shifted Test |
|---|---|---|---|
| No augmentation | 0.5167 ± 0.0189 | 0.4374 ± 0.0056 | 0.1840 ± 0.0059 |
| Geometric only | 0.5300 ± 0.0082 | 0.4258 ± 0.0097 | 0.2308 ± 0.0076 |
| Photometric only | 0.5367 ± 0.0047 | 0.4416 ± 0.0133 | 0.2010 ± 0.0317 |
| Geometric + photometric | 0.5100 ± 0.0082 | 0.4206 ± 0.0013 | 0.2499 ± 0.0069 |
| + Cutout / Random Erasing | 0.5000 ± 0.0163 | 0.4057 ± 0.0129 | 0.2662 ± 0.0077 |
| + Mixup + label smoothing | 0.5133 ± 0.0170 | 0.4248 ± 0.0046 | 0.2769 ± 0.0073 |
| **Final model** | 0.5133 ± 0.0170 | 0.4248 ± 0.0046 | **0.2769 ± 0.0073** |

## How to reproduce
1. Open `notebook.ipynb` on a GPU runtime (Kaggle or Colab) with internet enabled.
2. Run all cells from top to bottom. CIFAR-10 downloads automatically. Training all configurations takes a while.

## Notes
- The official hidden corruption was not provided. `shifted_test` uses only the listed corruptions, and Part E4 uses an **assumed** corruption (random grayscale, p = 0.5).
- Validation is the clean 100-image split, which favors weak augmentation.
- Parts E, F and E4 use seed 0.
