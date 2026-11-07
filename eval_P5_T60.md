1RUA24CSE0052, 1RUA24CSE0072, 1RUA24CSE0078, 1RUA24CSE0102

# Evaluation Results for P5 Fine-Grained Vehicle Recognition (Team T60)


## 1. Classification (Top-1 / Top-5)

| Approach | Top-1 | Top-5 | Make top-1 |
|---|---|---|---|
| **Classical (handcrafted + RBF SVM)** | **9.58%** | 28.27% | 18.28% |
| **Deep (ConvNeXt-Base @448 on YOLO crops)** | **94.89%** | 99.50% | 98.63% |
| **Hybrid (calibrated deep + gated classical fusion)** | **94.84%** | 99.47% | 98.63% |

Other variants (ablations, same test split):

| Variant | Family | Top-1 | Top-5 | Make top-1 |
|---|---|---|---|---|
| Classical, all except HOG features | classical | 1.37% | 4.43% | 3.56% |
| Classical, only HOG features | classical | 8.22% | 26.75% | 16.79% |
| Deep, 224 px input | deep | 93.04% | 99.10% | 97.79% |
| Deep + temperature scaling | deep | 94.89% | 99.50% | 98.63% |
| Deep + flip TTA | deep | 94.71% | 99.53% | 98.61% |
| Deep, GT crop (oracle detector) | deep | 94.88% | 99.50% | 98.63% |
| Deep, full image (no ROI crop) | deep | 94.06% | 99.32% | 97.96% |
| Deep, train + test on full images | deep | 95.01% | 99.61% | 98.76% |
| Deep, no auxiliary make head | deep | 94.88% | 99.35% | 98.57% |
| Deep, ensemble crop model + full-image model | deep | 95.50% | 99.61% | 98.83% |
| Hybrid (2-model ensemble) | hybrid | 95.47% | 99.64% | 98.86% |

**Best result: 2-model ensemble, 95.50% top-1 / 99.61% top-5.** The errors of the crop model and the full-image model are largely different (300 out of ~400 shared errors). Averaging them was preferred as it also outperforms the single model on the validation split (95.42% vs 95.09%). Deployed app uses the single crop model (half size and CPU time).

Chance: top-1 0.51%, top-5 2.55% (196 classes).

## 2. Object localization (IoU)

| Detector | Mean IoU | Median IoU | IoU ≥ 0.5 | IoU ≥ 0.75 |
|---|---|---|---|---|
| YOLO11s (fine-tuned on Stanford Cars) | 0.9545 | 0.9676 | 99.83% | 99.08% |
| YOLO11s COCO zero-shot (car+truck), baseline | 0.8857 | 0.9600 | 92.70% | 91.92% |

IoU between the highest confidence box of the detector and the single annotated box per image.

## 3. Object detection (mAP)

| Detector | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|
| YOLO11s (fine-tuned on Stanford Cars) | 0.9898 | 0.9372 |
| YOLO11s COCO zero-shot (car+truck), baseline | 0.8865 | 0.8213 |

Single class AP (COCO 101-point interpolation) for all detections (confidence ≥ 0.001); mAP@0.5:0.95 is average over IoU thresholds 0.50:0.05:0.95.

## 4. Model efficiency (size, inference time)

| Model | Size (MB) | Latency, mean (ms/image) | p95 (ms) | Hardware / precision |
|---|---|---|---|---|
| Classical (handcrafted + RBF SVM) | 38.5 | 332.2 | 462.0 | CPU (feature extraction 1 core) |
| Deep (ConvNeXt-Base @448 on YOLO crops) | 335.1 | 21.9 | 33.2 | NVIDIA GeForce RTX 4070 bf16 |
| Hybrid (calibrated deep + gated classical fusion) | 373.7 | 354.0 | 483.9 | NVIDIA GeForce RTX 4070 bf16 |
| YOLO11s (fine-tuned on Stanford Cars) | 18.3 | 11.1 | 12.4 | NVIDIA GeForce RTX 4070 fp16 |
| YOLO11s COCO zero-shot (car+truck), baseline | 18.4 | 11.3 | 11.7 | NVIDIA GeForce RTX 4070 fp16 |

Hybrid latency in practice: classical branch is needed only when calibrated deep confidence < 0.5 (1.8% of test images). Expected cost is 21.9 + 1.8% × 332 ≈ **27.9 ms/image** (above row assumes classical runs on all images).

Batch size 1. Deep/detector: RTX 4070 (GPU, bf16/fp16), including pre-/post-processing. Classical: parallel feature extraction (GrabCut dominates), 1 core per image.

**Optimization (deployed configuration): ONNX export + INT8 dynamic quantization, CPU, 2 threads**

| Variant | Size (MB) | Latency (ms/image) | Top-1 | Top-5 | n |
|---|---|---|---|---|---|
| Classifier (pytorch_fp32) | 335.1 | 540.8 | 94.80% | 99.40% | 500 |
| Classifier (onnx_fp32) | 335.0 | 489.6 | 94.80% | 99.40% | 500 |
| Classifier (onnx_int8_dynamic) | 95.0 | 215.9 | 95.20% | 99.40% | 500 |
| Detector (pytorch_fp32_cpu) | 18.3 | 47.1 | – | – | – |
| Detector (onnx_fp32_cpu) | 36.2 | 38.1 | – | – | – |

INT8: **3.5× smaller, 2.5× faster** on CPU. Top-1 change: +0.4 pts (on the same 500 test crops).

## 5. Robustness (performance in various conditions)

Classifier on a fixed random subset of **4000 test images**. Corrupt each detector crop, then classify.

| Condition | Top-1 | Δ vs clean | Top-5 | Mean confidence |
|---|---|---|---|---|
| clean | 94.90% | – | 99.65% | 0.955 |
| dark_gamma2.2 | 94.85% | -0.05 pts | 99.35% | 0.948 |
| bright_gamma0.45 | 94.65% | -0.25 pts | 99.62% | 0.952 |
| brightness-40 | 94.83% | -0.07 pts | 99.60% | 0.955 |
| brightness+40 | 94.62% | -0.27 pts | 99.58% | 0.952 |
| lowlight_noise | 92.42% | -2.47 pts | 99.10% | 0.922 |
| blur_sigma2 | 86.40% | -8.50 pts | 96.75% | 0.846 |
| occlusion10 | 94.38% | -0.52 pts | 99.40% | 0.946 |
| occlusion25 | 92.15% | -2.75 pts | 99.05% | 0.920 |
| occlusion50 | 78.67% | -16.23 pts | 93.15% | 0.805 |
| res224 | 93.15% | -1.75 pts | 99.25% | 0.923 |
| res128 | 86.90% | -8.00 pts | 98.05% | 0.820 |
| res64 | 37.48% | -57.42 pts | 67.07% | 0.502 |

Hybrid preprocessing: classical CLAHE + auto-gamma before CNN.

| Condition | Only CNN | + CLAHE/gamma | Δ |
|---|---|---|---|
| clean | 94.90% | 94.67% | -0.22 pts |
| dark_gamma2.2 | 94.85% | 94.73% | -0.12 pts |
| brightness-40 | 94.83% | 94.88% | +0.05 pts |
| lowlight_noise | 92.42% | 92.30% | -0.12 pts |

View angle (clean images, bucketed by detected-box aspect ratio):

| View | n | Top-1 |
|---|---|---|
| narrow (<1.3): front / rear views | 226 | 92.92% |
| medium (1.3-1.8): 3/4 views | 1532 | 95.56% |
| wide (>=1.8): side views | 2242 | 94.65% |

Detector on whole-image corruptions (500 test images):

| Condition | Mean IoU | IoU ≥ 0.5 |
|---|---|---|
| clean | 0.9535 | 99.60% |
| dark_gamma2.2 | 0.9482 | 99.40% |
| lowlight_noise | 0.9428 | 99.20% |
| blur_sigma2 | 0.9455 | 99.60% |
| occlusion25 | 0.8764 | 93.20% |
| occlusion50 | 0.5915 | 53.80% |
| res128 | 0.9467 | 100.00% |
| res64 | 0.8948 | 97.80% |

## 6. Confidence and thresholds

| Model | Mean confidence | Conf. when correct | Conf. when wrong | ECE (15 bins) |
|---|---|---|---|---|
| Deep (raw softmax) | 0.805 | 0.821 | 0.520 | 0.1435 |
| Deep + temperature scaling | 0.956 | 0.968 | 0.734 | 0.0162 |
| Hybrid (deployed) | 0.957 | 0.968 | 0.741 | 0.0151 |

T = 0.644 fitted on validation split (raw model is under-confident due to label smoothing, CutMix/MixUp).

Accept prediction if calibrated confidence ≥ τ (hybrid):

| τ | Coverage (answered) | Accuracy on answered | Error rate on rejected |
|---|---|---|---|
| 0.00 | 100.00% | 94.84% | – |
| 0.30 | 99.47% | 95.15% | 62.79% |
| 0.50 | 98.54% | 95.61% | 57.26% |
| 0.60 | 96.95% | 96.16% | 47.35% |
| 0.70 | 95.19% | 96.68% | 41.60% |
| 0.80 | 92.94% | 97.30% | 37.50% |
| 0.90 | 88.99% | 97.88% | 29.72% |
| 0.95 | 84.45% | 98.29% | 23.92% |

**Handling low confidence (deployed policy, thresholds on validation: τ_model = 0.85, τ_make = 0.9)**

| Answer given | Share of test images | Accuracy |
|---|---|---|
| Make + model + year | 91.16% | 97.57% |
| Manufacturer only | 4.20% | 99.70% (make) |
| "Uncertain" + top-5 shown | 4.64% | top-5 has true class: 90.35% |

Policy sweep (τ_make = 0.9 fixed):

| τ_model | Full answer: share / accuracy | Make only: share / accuracy | Uncertain share |
|---|---|---|---|
| 0.3 | 99.47% / 95.15% | 0.00% / – | 0.53% |
| 0.5 | 98.54% / 95.61% | 0.12% / 100.00% | 1.33% |
| 0.7 | 95.19% / 96.68% | 1.82% / 100.00% | 3.00% |
| 0.8 | 92.94% / 97.30% | 3.08% / 99.60% | 3.98% |
| 0.9 | 88.99% / 97.88% | 5.76% / 99.78% | 5.25% |
| 0.95 | 84.45% / 98.29% | 10.30% / 98.79% | 5.25% |

Plots: `results/hybrid/reliability_coverage.png` (reliability diagram, accuracy vs coverage), `results/robustness/robustness.png`. Ablations: `docs/ablation_study.md`. Failure cases: `docs/failure_analysis.md`.