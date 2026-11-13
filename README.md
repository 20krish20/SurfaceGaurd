## SurfaceGuard (PatchCore + WinCLIP + Streamlit)

[![CI](https://github.com/harshshri07/SurfaceGaurd/actions/workflows/ci.yml/badge.svg)](https://github.com/harshshri07/SurfaceGaurd/actions/workflows/ci.yml)

Industrial surface anomaly detection and localization with a deployable Streamlit UI.

### What you get
- **PatchCore**: train on normal images, detect + localize defects with anomaly heatmaps.
- **WinCLIP (practical variant)**: CLIP-based anomaly scoring with windowed heatmaps (zero/few-shot style).
- **Evaluation**: image AUROC, pixel AUROC, AU-PRO, F1.
- **UI**: side-by-side model comparison (PatchCore, WinCLIP, Hybrid) and benchmark runner.
- **Deployment**: **Hugging Face Docker Spaces** — live inference with uploads in the browser.

### Demo: Hugging Face Docker Space

This project is deployed as a **Hugging Face Space using Docker SDK** at: **https://huggingface.co/spaces/rishabh00/surfaceguard**

**PatchCore checkpoints (rubric “model”):** PatchCore is **not** a single PyTorch `.pt` file. Each **category** has **two** files that belong together: `memory.npz` (patch memory bank) and `meta.yaml` (settings + threshold), under `outputs/patchcore/<category>/`.

**Hugging Face Model repo layout:** upload the same tree you have locally, e.g. `patchcore/bottle/...`, `patchcore/cable/...`, for every category you trained. The app **syncs the whole `patchcore/**` tree** from the Hub when `outputs/patchcore` is empty, so **PatchCore “auto”** (pick best category after upload) works as before. Set **`hf_patchcore_repo`** in `configs/app.yaml` (or **`SURFACEGUARD_HF_REPO_ID`** on the Space). 

**WinCLIP (no separate trained file):** WinCLIP uses **OpenCLIP** pretrained weights (`ViT-B-32`, LAION). There is **no** large custom WinCLIP checkpoint like PatchCore’s `memory.npz`. Those weights **download automatically** when WinCLIP runs (cached after first use).

---

### Quick run (≤4 steps)

1. **Clone or unzip** this repository (code only in the zip).
2. **Install** dependencies and the local package:
   - **Linux / macOS:** `python3 -m venv .venv && source .venv/bin/activate`
   - **Windows:** `python -m venv .venv` then `.\.venv\Scripts\Activate.ps1`  
   Then: `pip install -U pip && pip install -r requirements.txt && pip install -e .`
3. **Model** — local `outputs/patchcore/<category>/` with `memory.npz` + `meta.yaml`, **or** set `hf_patchcore_repo` / `SURFACEGUARD_HF_REPO_ID` to sync `patchcore/**` from a Hub Model repo on first run (empty `outputs/patchcore`).
4. **One command to launch the UI** (latest PatchCore + WinCLIP as selected in the app):
   - **Linux / macOS:** `chmod +x scripts/run_project.sh && ./scripts/run_project.sh`
   - **Windows:** `.\scripts\run_project.ps1`

Alternatively, after step 2 (and model present or HF configured): `streamlit run app/Home.py`

Open the URL shown in the terminal.

### Folder layout
- `src/surfaceguard/`: library code
- `tools/`: CLI entrypoints (train/eval/infer)
- `configs/`: YAML configs
- `app/`: Streamlit UI
- `scripts/`: PowerShell helpers (Windows)
- `data/`: datasets (gitignored)
- `outputs/`: checkpoints, embeddings, reports (gitignored)

### Setup (Windows PowerShell)
```powershell
cd "C:\Users\shris\OneDrive\Desktop\SurfaceGaurd\SurfaceGaurd"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
pip install -e .
```

### Setup (macOS / Linux)
```bash
cd "/path/to/SurfaceGaurd"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
pip install -e .
```

### Dataset: MVTec AD
Download MVTec AD manually and place it like:

```
data/mvtec_ad/
  bottle/
    train/good/*.png
    test/good/*.png
    test/<defect_type>/*.png
    ground_truth/<defect_type>/*.png
```

Set your dataset root in `configs/patchcore_mvtec.yaml` and `configs/winclip_mvtec.yaml`.

### (Optional) Generate a tiny toy dataset
If you just want to verify the pipeline end-to-end without downloading MVTec yet:

```powershell
python tools\make_synth_mvtec.py --root data\mvtec_ad --category toy
```

### Run (CLI)
Train PatchCore for one category:
```powershell
python tools\train_patchcore.py --config configs\patchcore_mvtec.yaml --category bottle
```

Evaluate PatchCore:
```powershell
python tools\eval.py --config configs\patchcore_mvtec.yaml --method patchcore --categories all
```

Evaluate WinCLIP:
```powershell
python tools\eval.py --config configs\winclip_mvtec.yaml --method winclip --categories all
```

Benchmark all categories (writes per-category reports + an aggregated summary JSON under `outputs/reports/`):
```powershell
.\scripts\benchmark_all.ps1 -Method patchcore -Categories all
.\scripts\benchmark_all.ps1 -Method winclip   -Categories all
```

Single-image inference:
```powershell
python tools\infer.py --method patchcore --category bottle --image "path/to/image.png" --save_overlay
```

### Tests

The test suite is hermetic: it generates a synthetic dataset on the fly and swaps the pretrained backbone for a tiny random one, so it runs in seconds on a CPU and never downloads weights or data.

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest
```

What's covered: dataset loading and mask pairing, metrics (AUROC, F1, AU-PRO), PatchCore internals (patching, coreset selection, scoring), visualisation helpers, the CLIs, and an end-to-end **train → load → evaluate → infer** run. Tests marked `xfail` document known issues; delete the marker when the fix lands.

GitHub Actions (`.github/workflows/ci.yml`) runs a lint check for syntax errors and undefined names, then the tests on Python 3.10 (the Docker/HF Space version) and 3.12, on every push to `main` and every pull request.

### Run (UI)
```powershell
.\scripts\run_app.ps1
```

macOS / Linux:

```bash
streamlit run app/Home.py
```

### Run (PowerShell helper scripts)
These assume your venv is in `.\.venv\`:

```powershell
.\scripts\train_patchcore.ps1 -Category bottle
.\scripts\eval_patchcore.ps1  -Category bottle
.\scripts\eval_winclip.ps1    -Category bottle
```

### Trained PatchCore weights

Large **`memory.npz`** files are **gitignored** here so pushes stay small (GitHub / Hugging Face Spaces). After training locally you get `outputs/patchcore/<category>/`; upload that tree to a **Hugging Face Model** repo (see above) and set **`hf_patchcore_repo`** / **`SURFACEGUARD_HF_REPO_ID`** so the app downloads weights at runtime.

### Provenance
Training checkpoints (`outputs/patchcore/**/meta.yaml`) and evaluation reports (`outputs/reports/*.json`) include:
- `config_hash`: stable hash of the YAML config contents
- `git_commit`: short git commit id (when available)

### Hugging Face Docker Space (official deployment)

1. Create a **new Space** and choose **Docker** SDK.
2. Push this repository to that Space (or connect GitHub).
3. In Space **Settings → Variables and secrets**:
   - `SURFACEGUARD_HF_REPO_ID`: Model repo containing `patchcore/<category>/...` (optional but recommended)
   - `HF_TOKEN`: required only if that model repo is private
4. Build will use the root `Dockerfile` automatically; app serves on port `8501` (declared via README front matter `app_port`).
5. Open the Space URL and run inference from the UI.

