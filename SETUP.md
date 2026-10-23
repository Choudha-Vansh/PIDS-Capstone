# SETUP — ADAS Pedestrian Intent + AEB Simulator

Everything here runs **in software only** (no hardware). You need a machine with an
NVIDIA GPU for training (CPU works for the demo-video dashboard, just slower).

## 1. Environment

```bash
cd adas_sim
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install PyTorch matching YOUR CUDA version first (see https://pytorch.org):
# example (CUDA 12.1):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

## 2. Model weights (small, auto-downloadable)

```bash
python - <<'PY'
from ultralytics import YOLO
# downloads YOLO11-pose (or use yolov8n-pose.pt). Move the file into weights/.
YOLO("yolo11n-pose.pt")
PY
mkdir -p weights && mv yolo11n-pose.pt weights/ 2>/dev/null || true
```

BoT-SORT tracker config ships with `ultralytics` (`botsort.yaml`); no separate download.

## 3. Datasets

### JAAD (crossing intent, camera only) — you already have annotations
- Repo: https://github.com/ykotseruba/JAAD  (clone for the download script + split lists)
- Videos: run JAAD's `download_clips.sh`. Put clips under `datasets/JAAD/JAAD_clips/`
  and annotations under `datasets/JAAD/annotations/`.
- License: research use; see the JAAD repo.

### PIE (primary intent, has ego-vehicle speed) — needs registration
- Repo: https://github.com/aras62/PIE
- Register + download `PIE_clips` and annotations per the repo instructions.
- Put under `datasets/PIE/` matching the repo's expected layout.
- **This is the only dataset with ego-speed**, which real TTC needs — worth the setup.

### nuScenes (multimodal: camera + radar + LIDAR) — for the fusion path
- Site: https://www.nuscenes.org/nuscenes  (free account + EULA required)
- Start with **v1.0-mini** (~4 GB, 10 scenes) to build/validate fusion, then the full
  `v1.0-trainval` (~300+ GB) if you want to train the 3D side seriously.
- Extract under `datasets/nuscenes/` so `nuscenes-devkit` can load it.
- Set `paths.nuscenes_version` in `config/config.yaml` accordingly.

> If a dataset asks you to register / accept a EULA, do that yourself — the scripts
> here will not and should not bypass access controls.

## 4. Prepare training data (PIE + JAAD → sequences)

```bash
python data/prepare_jaad.py       # -> artifacts/jaad_sequences.npz
python data/prepare_pie.py        # -> artifacts/pie_sequences.npz
python data/combine.py            # -> artifacts/{X,y}_{train,val,test}.npy  (split by clip)
```

## 5. Train + evaluate the intent model

```bash
python intent/train.py            # trains temporal model, saves weights/intent_model.pth
python intent/calibrate.py        # temperature scaling on val, saves calibration
python intent/evaluate.py         # metrics on the held-out TEST split -> results/
```

## 6. Validate 3D fusion on nuScenes (optional, needs nuScenes)

```bash
python perception/fusion.py --demo   # runs fusion on a nuScenes-mini scene, prints 3D error
```

## 7. Run the dashboard on YOUR demo video

```bash
python run_demo.py --video path/to/demo.mp4
# camera path: pose -> tracking -> intent -> monocular distance -> TTC -> AEB -> dashboard
```

The dashboard replays the video with overlays and shows BRAKE/warning state, TTC,
distance, crossing probability, ego speed, per-stage latency and total system reaction
time, and writes a CSV log to `results/`.

## What still needs YOU (can't be automated from here)
- Downloading the datasets (registration/EULA + hundreds of GB for full nuScenes).
- GPU training runs (minutes–hours depending on dataset size and GPU).
- Providing the demo video for the final dashboard test.

## Phase gate
Only after the simulation metrics look good do we touch hardware. The sensor input is
already isolated behind `sensors/base.py`, so swapping simulated feeds for real
camera/radar/LIDAR + a Jetson target later is a localized change, not a rewrite.
