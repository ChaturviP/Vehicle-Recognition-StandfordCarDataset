# Runs the three retraining ablations back to back (about 2-2.5 h on an RTX 4070).
# Usage (venv active, project root):  powershell -ExecutionPolicy Bypass -File scripts\run_retrain_ablations.ps1
# Each run: train -> predict on test (detector crops, or full images for the full-image model) -> evaluate.
$ErrorActionPreference = "Stop"
$det = "results/preds/detector_yolo11s_cars.npz"

function Run-Ablation($name, $trainArgs, $crop) {
    Write-Host "=== $name ===" -ForegroundColor Cyan
    python -m src.deep.train_classifier --name $name @trainArgs
    if ($LASTEXITCODE -ne 0) { throw "training $name failed" }
    if ($crop -eq "det") {
        python -m src.deep.predict_classifier --ckpt "models/classifier_$name.pt" --crop det --det-preds $det
    } else {
        python -m src.deep.predict_classifier --ckpt "models/classifier_$name.pt" --crop none
    }
    python -m src.eval.evaluate "results/preds/${name}_${crop}crop.npz"
}

Run-Ablation "convnext_224" @("--img", "224", "--batch", "32", "--accum", "1") "det"
Run-Ablation "convnext_nomake" @("--make-weight", "0") "det"
Run-Ablation "convnext_fullimg" @("--crop", "none") "none"
python -m src.eval.ablation
Write-Host "All retraining ablations done." -ForegroundColor Green
