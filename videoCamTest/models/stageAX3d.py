# at top
import torch, torch.nn as nn, numpy as np
try:
    from pytorchvideo.models.hub import x3d_m
except Exception as e:
    raise RuntimeError(
        "pytorchvideo is required for x3d_m. Please install 'pytorchvideo' and 'torch'."
    ) from e

def normalizeClipCthw(clip, mean=(0.45, 0.45, 0.45), std=(0.225, 0.225, 0.225)):
    meanT = torch.tensor(mean, dtype=clip.dtype, device=clip.device)[:, None, None, None]
    stdT  = torch.tensor(std,  dtype=clip.dtype, device=clip.device)[:, None, None, None]
    return (clip - meanT) / stdT

def adaptAndLoad(model, head, ckpt):
    # Normalize to flat dict of {name: tensor}
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        ckpt = ckpt["state_dict"]

    # Unwrap common prefixes (nn.DataParallel or model.)
    def strip_prefix(k):
        for p in ("module.", "model."):
            if k.startswith(p):
                return k[len(p):]
        return k
    ckpt = {strip_prefix(k): v for k, v in ckpt.items()}

    # If dict is nested with backbone/head keys
    if "backbone" in ckpt and isinstance(ckpt["backbone"], dict):
        flat = {}
        for k, v in ckpt["backbone"].items():
            flat[f"backbone.{k}"] = v
        if "head" in ckpt and isinstance(ckpt["head"], dict):
            for k, v in ckpt["head"].items():
                nk = k[3:] if k.startswith("fc.") else k
                flat[f"head.{nk}"] = v
        ckpt = flat

    # If looks like raw x3d keys (e.g., 'blocks.*'), treat them as backbone
    if not any(k.startswith("backbone.") or k.startswith("head.") for k in ckpt.keys()):
        if any(k.startswith("blocks.") for k in ckpt.keys()):
            ckpt = {f"backbone.{k}": v for k, v in ckpt.items()}
        else:
            # fallback: assume all go to backbone
            ckpt = {f"backbone.{k}": v for k, v in ckpt.items()}

    # split for backbone/head
    bb = {k.replace("backbone.", "", 1): v for k, v in ckpt.items() if k.startswith("backbone.")}
    hd = {k.replace("head.", "", 1): v for k, v in ckpt.items() if k.startswith("head.")}

    msd = model.state_dict()
    hsd = head.state_dict()
    loaded_bb = 0
    loaded_hd = 0
    # backbone
    for k, v in bb.items():
        if k in msd and msd[k].shape == v.shape:
            msd[k] = v; loaded_bb += 1
    # head
    for k, v in hd.items():
        if k in hsd and hsd[k].shape == v.shape:
            hsd[k] = v; loaded_hd += 1
    model.load_state_dict(msd, strict=False)
    head.load_state_dict(hsd, strict=False)
    total = len(msd) + len(hsd)
    return (loaded_bb, loaded_hd), total

class StageAX3d(nn.Module):
    def __init__(self, device: str = "cuda:0", checkpointPath: str = "", invertOutput: bool = False,
                 normalize: str | None = None, mean: tuple | None = None, std: tuple | None = None, skipNorm: bool = False):
        super().__init__()
        # avoid network downloads; weights will be loaded from checkpoint
        self.backbone = x3d_m(pretrained=False)
        inDim = self.backbone.blocks[-1].proj.in_features
        self.backbone.blocks[-1].proj = nn.Identity()
        self.head = nn.Linear(inDim, 1)
        self.invertOutput = invertOutput
        self.skipNorm = skipNorm
        # choose normalization
        if mean is not None and std is not None:
            self.normMean, self.normStd = tuple(mean), tuple(std)
        elif normalize == "imagenet":
            self.normMean, self.normStd = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        else:  # default kinetics-like
            self.normMean, self.normStd = (0.45, 0.45, 0.45), (0.225, 0.225, 0.225)

        if checkpointPath and checkpointPath.strip():
            sd = torch.load(checkpointPath, map_location="cpu")
            # unwrap common containers
            if isinstance(sd, dict) and any(k in sd for k in ("model","state_dict","weights")):
                sd = sd.get("model", sd.get("state_dict", sd.get("weights", sd)))
            (lbb, lhd), total = adaptAndLoad(self.backbone, self.head, sd)
            # quick parameter norms to check non-zero weights
            with torch.no_grad():
                wnorm = float(self.head.weight.detach().norm().cpu())
                bnorm = float(self.head.bias.detach().norm().cpu())
            print(f"[StageA] loaded backbone={lbb} head={lhd} approxTotal={total} head|w_norm={wnorm:.3f} b_norm={bnorm:.3f}")
        else:
            print("[StageA] WARNING: running without fine-tuned checkpoint")

        # device fallback
        want = device
        if device.startswith("cuda") and not torch.cuda.is_available():
            print(f"[StageA] CUDA not available; falling back to CPU from {device}")
            want = "cpu"
        self.to(want).eval()
        self.device = want
        self._dbg_calls = 0

    @torch.no_grad()
    def forwardClip(self, clipNdarray):  # T x H x W x C (BGR uint8)
        if isinstance(clipNdarray, list):
            clipNdarray = np.stack(clipNdarray, axis=0)
        # BGR->RGB, [0,1]
        clip = clipNdarray[:, :, :, ::-1].astype(np.float32) / 255.0
        clip = torch.from_numpy(clip).permute(3,0,1,2).contiguous().to(self.device)  # C,T,H,W
        if not self.skipNorm:
            clip = normalizeClipCthw(clip, self.normMean, self.normStd)
        feats = self.backbone(clip.unsqueeze(0))  # B=1
        logit = self.head(feats)
        prob = torch.sigmoid(logit).item()
        if self.invertOutput:
            prob = 1.0 - prob
        # lightweight one-time debug to ensure variations
        if self._dbg_calls < 5:
            with torch.no_grad():
                m = float(feats.detach().mean().cpu()) if isinstance(feats, torch.Tensor) else 0.0
                s = float(feats.detach().std().cpu()) if isinstance(feats, torch.Tensor) else 0.0
            print(f"[StageA] debug feats mean={m:.4f} std={s:.4f} prob={prob:.4f}")
            self._dbg_calls += 1
        return float(prob)
