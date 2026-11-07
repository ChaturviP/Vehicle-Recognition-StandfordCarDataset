# Ablation Study

Each row of the table below removes or changes one component of the system and measures its effect on the official Stanford Cars **test split (8,041 unseen images)**, unless otherwise stated. Δ is the change against the row's reference. Raw numbers for every run are in `results/metrics/`.

## 1. Detection and region of interest

Reference: full system, YOLO11s crop → ConvNeXt-Base @448 (**94.89%** top-1, ECE 0.144).

| Variant | What changed | Top-1 | Δ | ECE |
|---|---|---|---|---|
| Ground-truth box instead of YOLO's box | detector replaced by a perfect one | 94.88% | −0.01 | 0.143 |
| No crop at test time (full image) | detection and cropping removed | 94.06% | −0.83 | 0.177 |
| Network trained and tested on full images | cropping removed everywhere | 95.01% | +0.12 | 0.127 |

| Detector variant | Mean IoU | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|
| YOLO11s fine-tuned on Stanford Cars (used) | 0.954 | 0.990 | 0.937 |
| YOLO11s pre-trained on COCO, no fine-tuning | 0.886 | 0.887 | 0.821 |

**Findings**
* **The detector is not a bottleneck.** A perfect box gives the same accuracy as YOLO's box.
* **Cropping matters for a network trained on crops**: skipping it costs 0.83 points and
  worsens calibration. A network trained on full images does just as well without a crop
  (+0.12, within noise), which is why the two make a good ensemble (section 2).
* **Fine-tuning the detector adds 11.6 points of mAP@0.5:0.95.** The pre-trained model often
  boxes a background car instead of the photo's subject.

## 2. Classifier

| Variant | What changed | Top-1 | Δ | ECE |
|---|---|---|---|---|
| Input 224 px instead of 448 px | resolution halved | 93.04% | −1.85 | 0.140 |
| No auxiliary manufacturer head | hierarchical loss removed | 94.88% | −0.01 | 0.156 |
| Flip test-time augmentation | horizontal-flip TTA added | 94.71% | −0.17 | 0.141 |
| Ensemble: crop network + full-image network | two networks averaged | **95.50%** | **+0.61** | 0.143 |

**Findings**
* **Resolution is the most important setting.** Fine-grained cues (badges, grilles, lamps) need
  448 px.
* **The manufacturer head does not change accuracy**, but it worsens calibration when removed
  (ECE 0.144 → 0.156). It also powers the manufacturer-only fallback in the app.
* **Flip TTA slightly hurts** because cars are not left–right symmetric (badges, fuel caps).
* **The ensemble is the best configuration.** The two networks share only 300 of their ~400
  errors each. It was adopted only after it also won on the validation split (95.42% vs 95.09%).

## 3. Hybrid integration

| Variant | What changed | Top-1 | Δ | ECE |
|---|---|---|---|---|
| Deep network, raw confidence | reference | 94.89% | – | 0.144 |
| + temperature scaling (T = 0.644) | calibration added | 94.89% | 0.00 | 0.016 |
| + gated classical score fusion (deployed hybrid) | SVM scores added when the network is unsure | 94.84% | −0.05 | **0.015** |
| Hybrid applied to the two-network ensemble | ensemble + calibration + fusion | 95.47% | +0.58 | **0.015** |

Classical CLAHE + gamma applied before the network, under difficult lighting (4,000 test images):

| Condition | Network only | + CLAHE / gamma | Δ |
|---|---|---|---|
| Clean | 94.90% | 94.67% | −0.22 |
| Dark (gamma 2.2) | 94.85% | 94.73% | −0.12 |
| Brightness −40% | 94.83% | 94.88% | +0.05 |
| Low light + sensor noise | 92.42% | 92.30% | −0.12 |

**Findings**
* **Calibration is the most valuable hybrid component.** It cuts the calibration error by 9×
  at no accuracy cost, which makes the confidence thresholds meaningful (91.2% of images get a
  full answer at 97.6% accuracy).
* **Fusing the classical scores is accuracy-neutral** (−0.05, four images) but slightly improves
  calibration. The SVM is right on only 41 of the network's 411 mistakes.
* **Classical lighting correction does not help the network**, because colour-jitter and
  RandAugment training already made it lighting-invariant. It does help the classical branch
  itself.

## 4. Classical branch

| Variant | Features used | Top-1 | Δ | Top-5 |
|---|---|---|---|---|
| Full classical pipeline (reference) | all 7 groups, 3,911 values | 9.58% | – | 28.27% |
| HOG only | shape/gradient only | 8.22% | −1.36 | 26.75% |
| Everything except HOG | colour, LBP, GLCM, edges, contour shape, geometry | 1.37% | −8.21 | 4.43% |

Chance level is 0.51% top-1 (196 classes).

**Findings**
* **HOG carries most of the classical signal**: it captures the silhouette and panel lines.
* **The other groups are weak on their own**, but they add 1.4 points on top of HOG. Colour is
  nearly useless for identity, since any model comes in any colour.
* **Handcrafted features reach about 19× chance** but are no match for the network (94.9%).
  The value of the classical branch is interpretability (the processing stages shown in the app)
  and a small calibration gain, not accuracy.

## Summary

| Component | Effect when present |
|---|---|
| Input resolution 448 px (vs 224) | +1.85 points top-1 |
| Second network (ensemble) | +0.61 points top-1 |
| ROI crop (for the crop-trained network) | +0.83 points top-1, better calibration |
| Detector fine-tuning | +11.6 points mAP@0.5:0.95 |
| Temperature scaling | ECE 0.144 → 0.016 |
| Classical score fusion | ECE 0.016 → 0.015, accuracy neutral |
| Manufacturer head | accuracy neutral, better calibration, enables make-level fallback |
| Flip TTA | −0.17 points (left off) |
| Classical CLAHE before the network | no effect (left off) |