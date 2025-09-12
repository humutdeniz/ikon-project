import torch, torch.nn as nn
import numpy as np
try:
    from pytorchvideo.models.hub import slowfast_r50
except Exception as e:
    raise RuntimeError(
        "pytorchvideo is required for slowfast_r50. Please install 'pytorchvideo' and 'torch'."
    ) from e

class StageBSlowfast(nn.Module):
    def __init__(self, device: str = "cuda:1", checkpointPath: str = ""):
        super().__init__()
        # avoid network downloads; weights will be loaded from checkpoint if provided
        self.backbone = slowfast_r50(pretrained=False)  # outputs 400-way by default
        # Replace head with binary; SlowFast returns logits from 'blocks.6.proj'
        inDim = self.backbone.blocks[-1].proj.in_features
        self.backbone.blocks[-1].proj = nn.Identity()
        self.head = nn.Linear(inDim, 1)
        if checkpointPath:
            sd = torch.load(checkpointPath, map_location="cpu")
            # allow loading either full model or just head/backbone combos
            if isinstance(sd, dict) and any(k in sd for k in ("model","state_dict","weights")):
                sd = sd.get("model", sd.get("state_dict", sd.get("weights", sd)))
            self.load_state_dict(sd, strict=False)
        want = device
        if device.startswith("cuda") and not torch.cuda.is_available():
            print(f"[StageB] CUDA not available; falling back to CPU from {device}")
            want = "cpu"
        self.to(want)
        self.eval()
        self.device = want

    @torch.no_grad()
    def forwardClip(self, clipNdarray):  # T x H x W x C (BGR)
        # SlowFast expects a list [slow_path, fast_path], both CTHW
        clip = self._toTensor(clipNdarray)  # C,T,H,W
        fastPath = clip
        # Slow path = subsample by 4 (alpha=4)
        slowPath = clip[:, ::4, :, :]
        inp = [slowPath.unsqueeze(0).to(self.device), fastPath.unsqueeze(0).to(self.device)]
        feats = self.backbone(inp)
        logit = self.head(feats)
        prob = torch.sigmoid(logit).item()
        return prob

    def normalizeClipCthw(clip, mean=(0.45,0.45,0.45), std=(0.225,0.225,0.225)):
        # clip: torch.FloatTensor [C,T,H,W] in [0,1]
        mean = torch.tensor(mean, dtype=clip.dtype, device=clip.device)[:, None, None, None]
        std  = torch.tensor(std,  dtype=clip.dtype, device=clip.device)[:, None, None, None]
        return (clip - mean) / std

    def _toTensor(self, clip):
        if isinstance(clip, list):
            clip = np.stack(clip, axis=0)
        clip = clip[:, :, :, ::-1].astype(np.float32) / 255.0
        clip = torch.from_numpy(clip).permute(3,0,1,2).contiguous()  # CTHW
        clip = self.normalizeClipCthw(clip)
        return clip
