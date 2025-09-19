# infer_meta.py
import argparse, json
from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from torchvision import transforms
from torchvision import models

ImageFile.LOAD_TRUNCATED_IMAGES = True

def buildEvalTf(mean, std, imgSize):
    return transforms.Compose([
        transforms.Resize(int(imgSize * 1.15)),
        transforms.CenterCrop(imgSize),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

def create_model(arch: str, num_classes: int):
    arch = (arch or "mbv3_small").lower()
    if arch in ("mbv3_small", "mobilenet_v3_small"):
        m = models.mobilenet_v3_small(weights=None)
        inFeat = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(inFeat, num_classes)
        return m
    if arch in ("mbv3_large", "mobilenet_v3_large"):
        m = models.mobilenet_v3_large(weights=None)
        inFeat = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(inFeat, num_classes)
        return m
    if arch in ("effnet_b3", "efficientnet_b3"):
        m = models.efficientnet_b3(weights=None)
        inFeat = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(inFeat, num_classes)
        return m
    if arch in ("resnet50",):
        m = models.resnet50(weights=None)
        inFeat = m.fc.in_features
        m.fc = nn.Linear(inFeat, num_classes)
        return m
    raise ValueError(f"Unknown arch: {arch}")

def loadModel(modelPath: Path, numClasses: int, arch: str):
    model = create_model(arch, numClasses)
    sd = torch.load(modelPath, map_location="cpu")
    model.load_state_dict(sd)
    model.eval()
    return model


def main():
    path = input()
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", type=str, default="models/meta.json")
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--image", type=str, default=path)
    ap.add_argument("--threshold", type=float, default=0.8)
    args = ap.parse_args()

    # load meta.json
    with open(args.meta, "r", encoding="utf-8") as f:
        meta = json.load(f)
    classes = meta["classes"]
    mean, std = meta["mean"], meta["std"]
    imgSize = meta["img_size"]
    arch = meta.get("arch", "mbv3_small")

    # resolve model path: prefer CLI, else infer from arch
    model_path = Path(args.model) if args.model else Path("models") / f"violence_{arch}_best.pt"
    model = loadModel(model_path, numClasses=len(classes), arch=arch)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    tfm = buildEvalTf(mean, std, imgSize)
    img = Image.open(args.image).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu()

    violenceIdx = meta["class_to_idx"]["violence"]
    pViolence = float(probs[violenceIdx])
    predIdx = int(probs.argmax().item())
    predName = classes[predIdx]

    print({
        "pred": predName,
        "pViolence": round(pViolence, 4),
        "threshold": args.threshold,
        "alert": pViolence >= args.threshold,
        "classes": classes
    })

if __name__ == "__main__":
    main()
