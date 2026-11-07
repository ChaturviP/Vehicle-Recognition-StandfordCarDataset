# Intelligent Fine-Grained Vehicle Recognition and Analysis System (P5)

**Live app:** [vehicle-recognition-standfordcardataset-team-altf4.streamlit.app](https://vehicle-recognition-standfordcardataset-team-altf4.streamlit.app/) · **Evaluation:** [`eval_P5_T60.md`](eval_P5_T60.md)
**Demo Video:** https://drive.google.com/drive/folders/1C13M2CYJiMd7Q8XSDy65zAkOFW9z2glF?usp=sharing

**Team T60 — alt f4**

| Member | USN |
|---|---|
| Aniketh K Menon | 1RUA24CSE0052 |
| Aryav Agrawal | 1RUA24CSE0072 |
| Ayush S Kulkarni | 1RUA24CSE0078 |
| Chaturvi P | 1RUA24CSE0102 |

| Deliverable | Location |
|---|---|
| Live hosted application | https://vehicle-recognition-standfordcardataset-team-altf4.streamlit.app/ |
| Complete source code | `src/` (data, detection, classical, deep, hybrid, evaluation, optimisation), `app/` |
| System architecture diagram | [`docs/architecture.png`](docs/architecture.png) |
| Design document | [`docs/design.md`](docs/design.md) |
| Comparative evaluation results | [`eval_P5_T60.md`](eval_P5_T60.md) |
| Ablation study | [`docs/ablation_study.md`](docs/ablation_study.md) |
| Solution report | [`docs/Report_P5_T60_altf4.pdf`](docs/Report_P5_T60_altf4.pdf) |
| Failure-case analysis | [`docs/failure_analysis.md`](docs/failure_analysis.md) |

Detects a car in an image, draws its bounding box, and predicts its **make, model and
year** (196 Stanford Cars classes) with a confidence score. Three pipelines are built
and compared: **classical CV**, **deep learning**, and a **hybrid** of the two.

> Status: complete. Every component is built, evaluated on the unseen test split and deployed.
> Design document: `docs/design.md` · Architecture: `docs/architecture.png` · Failure analysis:
> `docs/failure_analysis.md` · Ablations: `docs/ablation_study.md`.

## Setup (Windows, RTX GPU)

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python scripts/check_env.py
```

`check_env.py` should report the GPU and a passing matmul test.

## Data

Uses the Kaggle "Stanford Car Dataset by classes folder" mirror
(`anno_train.csv`, `anno_test.csv`, `names.csv`, `car_data/car_data/{train,test}/<class>/`).

```powershell
python -m src.data.prepare --root "C:/Users/win11/Downloads/archive (10)"
python -m src.detect.prepare_yolo
```

This writes `data/processed/index.csv` (one row per image: split, path, bbox, image size,
class, make, model, year, body type) and a YOLO-format copy in `data/yolo/`.
Splits: **train 6,922 / val 1,222** (stratified 15% of the official train set) and
**test 8,041** (official test set, used only for final evaluation).

## Step 2 — vehicle detector

```powershell
python -m src.detect.train_yolo                                   # YOLO11s, 30 epochs, ~30-60 min on an RTX 4070
python -m src.detect.evaluate_yolo --weights models/detector_yolo11s_cars.pt
python -m src.detect.evaluate_yolo --weights yolo11s.pt --coco --name yolo11s_coco_zeroshot
python -m src.eval.evaluate results/preds/detector_yolo11s_cars.npz results/preds/yolo11s_coco_zeroshot.npz
```

The zero-shot COCO run (car + truck classes, no fine-tuning) is the baseline row for the
detection comparison.

## Step 3 — fine-grained classifier

```powershell
python -m src.deep.train_classifier                    # ConvNeXt-Base @448 on GT crops, 30 epochs
python -m src.deep.predict_classifier --ckpt models/classifier_convnext_base_448.pt --crop det --det-preds results/preds/detector_yolo11s_cars.npz
python -m src.deep.predict_classifier --ckpt models/classifier_convnext_base_448.pt --crop gt --split val
```

Training uses ground-truth crops (padded 12%), label smoothing, CutMix/MixUp, RandAugment,
random erasing, and an auxiliary make head. The best epoch by validation top-1 is saved to
`models/classifier_<name>.pt`. Test predictions use the detector's boxes (`--crop det`), so they
measure the full system.

## Step 4 — classical branch

```powershell
python -m src.classical.extract_features
python -m src.classical.train_classical
python -m src.classical.show_stages --n 4
```

Preprocessing (CLAHE on LAB-L, auto-gamma, bilateral filter) → ROI crop → GrabCut mask →
Canny edges and contours → features: HSV colour histograms, LBP, GLCM, HOG, edge density,
contour shape (solidity, extent, Hu moments), box geometry → standardise → PCA → RBF SVM.
Ablation switches: `--no-clahe`, `--no-gamma`, `--no-grabcut` (extraction) and `--groups` (training).

## Step 5 — hybrid fusion and confidence handling

```powershell
python -m src.hybrid.build_hybrid
```

Fits temperature scaling on validation, chooses a deep + classical score-fusion weight on
validation NLL, and sets a three-level answer policy: full make/model/year when confident,
manufacturer only when only the make is certain, otherwise "uncertain" with the top-5.

## Step 6 — robustness, ablations, optimization

```powershell
python -m src.eval.robustness --ckpt models/classifier_convnext_base_448.pt --detector models/detector_yolo11s_cars.pt
python -m src.optimize.export_onnx --ckpt models/classifier_convnext_base_448.pt --detector models/detector_yolo11s_cars.pt
python -m src.eval.ablation
```

Robustness: illumination, low light + noise, blur, occlusion and resolution corruptions on a fixed
4,000-image test subset, with and without classical CLAHE enhancement in front of the CNN, plus
accuracy by viewpoint (box aspect ratio). Optimization: ONNX FP32 and INT8 (dynamic quantization)
benchmarked on 2 CPU threads. `src/eval/ablation.py` collects the raw ablation runs (summarised in `docs/ablation_study.md`).

## Step 7 — app and deployment

```powershell
pip install -r app/requirements.txt
streamlit run app/streamlit_app.py                 # local: http://localhost:8501
```

The deployed app runs from this repository on **Streamlit Community Cloud**: main file
`app/streamlit_app.py`, Python 3.12. The cloud installs `app/requirements.txt` (light, CPU only),
not the training requirements. The four files it needs are committed in `models/`:
`detector_yolo11s_cars.onnx`, `convnext_base_448_int8.onnx`, `classical_svm_full.joblib` and
`hybrid_config.json`. Thresholds are fixed to the values selected on validation.

The app runs on CPU with ONNX Runtime: YOLO11s detector → padded crop → ConvNeXt-Base INT8 plus the
classical pipeline → calibrated hybrid decision (full answer / make only / uncertain + top-5). It shows
the box and label, the confidence, the top-5, what each approach (classical, deep, hybrid) predicted,
and the classical processing stages, plus tabs with the test-set comparison and plots.
`app/app.py` is an equivalent Gradio front end.

## Evaluation harness

Every pipeline saves predictions with `src.eval.evaluate.save_predictions(...)`, then:

```powershell
python -m src.eval.evaluate results/preds/<run>.npz
python -m src.eval.report --team T01 --usns "USN1,USN2,USN3"
```

The report writes `eval_P5_<TeamID>.md` with metrics in the required order:
Top-1/Top-5 → IoU → mAP → size/latency → robustness → confidence/threshold analysis.

## Repository layout

```
src/data/       index building, label parsing, PyTorch dataset with bbox crops
src/detect/     YOLO export / training / inference
src/classical/  preprocessing, handcrafted features, SVM
src/deep/       CNN / ViT fine-tuning
src/hybrid/     fusion and low-confidence handling
src/eval/       metrics, evaluate.py, report.py
src/optimize/   ONNX export, INT8 quantization, pruning
app/            Gradio app
docs/           architecture diagram, design doc, ablation and failure analysis
results/        metrics JSON and plots
tests/          unit tests (pytest)
```
