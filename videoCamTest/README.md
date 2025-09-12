ViolenceGuard — Live Camera Test

Overview

- Opens a camera or video source and runs a 2‑stage pipeline (Stage A: X3D‑M, Stage B: SlowFast optional) for violence detection.
- Uses the weights in `weights/x3d_finetuned_packed.pth` for Stage A.
- Displays an overlay with current status (ALERT/SAFE) and the latest probabilities.

Prerequisites

- Python 3.9+
- Packages: `torch`, `pytorchvideo`, `opencv-python`, `pyyaml`, `numpy`
- Optional GPU. The code falls back to CPU if CUDA is unavailable.

Run

1) Ensure the weights file exists: `videoCamTest/weights/x3d_finetuned_packed.pth`
2) From this folder:

   ```bash
   cd ikon-project/videoCamTest
   python mainLive.py --config configs/defaultConfig.yaml
   ```

Config Notes (`configs/defaultConfig.yaml`)

- `videoSource`: `0` for default webcam, or a path/RTSP URL.
- `inputSize`, `clipSeconds`, `sampleFps`: control clip length and spatial size. Default produces 16 frames at 224x224.
- `stageA.device`: set to `cuda:0` for GPU, or `cpu`. The code automatically falls back to CPU if CUDA isn’t available.
- `useDisplay`: set `true` to show a window. Press `Esc` or `q` to quit.

Troubleshooting

- If `pytorchvideo` is missing, install it along with a matching PyTorch build.
- If the X3D weights fail to load, verify the path in the config and that the checkpoint matches the X3D‑M backbone + 1‑way head.

