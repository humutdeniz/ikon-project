"""Fine-tune an image classifier on the violence dataset.
Supports multiple backbones (MobileNetV3 Small/Large, EfficientNet-B3, ResNet50).
Robust to missing CUDA, corrupt images, and CWD differences.
"""
import os, time, random, json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.distributed as dist
from torchvision import datasets, transforms, models
from torchvision.models.mobilenetv3 import (
    MobileNet_V3_Small_Weights,
    MobileNet_V3_Large_Weights,
)
from torchvision.models import (
    EfficientNet_B3_Weights,
    ResNet50_Weights,
)
from PIL import Image, ImageFile

# tolerate slightly truncated JPEGs instead of crashing
ImageFile.LOAD_TRUNCATED_IMAGES = True

# -------------------------- config --------------------------
ROOT = Path(__file__).parent
dataDir = ROOT / "data"       # expects data/train, data/val (or validation), data/test
saveDir = ROOT / "models"
saveDir.mkdir(parents=True, exist_ok=True)

arch = os.environ.get("ARCH", "mobilenet_v3_large").lower() 
usePretrained = os.environ.get("USE_PRETRAINED", "0").lower() in ("1", "true", "yes")
freezeBackbone = os.environ.get("FREEZE_BACKBONE", "0").lower() in ("1", "true", "yes")
imgSize = int(os.environ.get("IMG_SIZE_OVERRIDE", 224))            # overridden for some backbones below
batchSize = int(os.environ.get("BATCH_SIZE", 64))
epochs = int(os.environ.get("EPOCHS", 12))
baseLr = float(os.environ.get("LR", 3e-4))
weightDecay = float(os.environ.get("WEIGHT_DECAY", 1e-4))
numWorkers = int(os.environ.get("NUM_WORKERS", 0))  # start with 0; override via env
seed = 1337
# ------------------------------------------------------------

def setSeed(seedVal: int):
    random.seed(seedVal)
    torch.manual_seed(seedVal)
    torch.cuda.manual_seed_all(seedVal)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def buildTransforms(weights, img_size):
    # eval: use the official weights transform if available (includes normalize)
    if weights is not None:
        evalTfms = weights.transforms()
        # extract mean/std from eval pipeline; fall back to ImageNet
        mean, std = IMAGENET_MEAN, IMAGENET_STD
        try:
            for t in getattr(evalTfms, "transforms", []):
                if isinstance(t, transforms.Normalize):
                    mean, std = t.mean, t.std
                    break
        except Exception:
            pass
    else:
        evalTfms = transforms.Compose([
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])
        mean, std = IMAGENET_MEAN, IMAGENET_STD

    # train: your augments + the same normalization
    trainTfms = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return trainTfms, evalTfms, mean, std


def _verify_and_filter(dataset: datasets.ImageFolder, name: str, step: int = 500):
    n = len(dataset.samples)
    if n == 0:
        print(f"[warn] {name}: 0 images found")
        return
    print(f"[verify] {name}: checking {n} images…")
    ok, bad = [], []
    for i, (path, target) in enumerate(dataset.samples, 1):
        try:
            with Image.open(path) as im:
                im.verify()
            ok.append((path, target))
        except Exception:
            bad.append(path)
        if i % step == 0 or i == n:
            print(f"  {name}: {i}/{n}")
    if bad:
        print(f"[warn] {name}: dropped {len(bad)} corrupt/unreadable images")
    dataset.samples = ok
    dataset.imgs = ok

def buildDatasetsAndLoaders(dataRoot: Path, trainTfms, evalTfms, distributed: bool = False):
    val_name = "validation" if (dataRoot / "validation").exists() else "val"
    missing = [p for p in [dataRoot / "train", dataRoot / val_name, dataRoot / "test"] if not p.exists()]
    if missing:
        print("[error] Missing dataset splits:")
        for p in missing:
            print(" -", p)
        raise SystemExit(1)

    trainSet = datasets.ImageFolder(dataRoot / "train", transform=trainTfms)
    valSet   = datasets.ImageFolder(dataRoot / val_name, transform=evalTfms)
    testSet  = datasets.ImageFolder(dataRoot / "test", transform=evalTfms)

    if os.environ.get("SKIP_VERIFY", "").lower() in ("1", "true", "yes"):
        print("[info] SKIP_VERIFY set; skipping image verification")
    else:
        _verify_and_filter(trainSet, "train")
        _verify_and_filter(valSet,   val_name)
        _verify_and_filter(testSet,  "test")

    pin_mem = torch.cuda.is_available()

    if distributed and dist.is_available() and dist.is_initialized():
        # Shard datasets across processes
        trainSampler = torch.utils.data.distributed.DistributedSampler(trainSet, shuffle=True)
        valSampler   = torch.utils.data.distributed.DistributedSampler(valSet,   shuffle=False)
        testSampler  = torch.utils.data.distributed.DistributedSampler(testSet,  shuffle=False)

        trainLoader = DataLoader(trainSet, batchSize, shuffle=False, sampler=trainSampler,
                                 num_workers=numWorkers, pin_memory=pin_mem)
        valLoader   = DataLoader(valSet,   batchSize, shuffle=False, sampler=valSampler,
                                 num_workers=numWorkers, pin_memory=pin_mem)
        testLoader  = DataLoader(testSet,  batchSize, shuffle=False, sampler=testSampler,
                                 num_workers=numWorkers, pin_memory=pin_mem)
    else:
        trainLoader = DataLoader(trainSet, batchSize, shuffle=True,  num_workers=numWorkers, pin_memory=pin_mem)
        valLoader   = DataLoader(valSet,   batchSize, shuffle=False, num_workers=numWorkers, pin_memory=pin_mem)
        testLoader  = DataLoader(testSet,  batchSize, shuffle=False, num_workers=numWorkers, pin_memory=pin_mem)

    return (trainSet, valSet, testSet), (trainLoader, valLoader, testLoader)

def buildModel(numClasses: int, usePretrainedFlag: bool, freeze: bool, arch: str):
    arch = arch.lower()
    weights = None
    model = None
    head_attr = None  # (object, attribute name or tuple path)

    if arch in ("mbv3_small", "mobilenet_v3_small"):
        weights = MobileNet_V3_Small_Weights.DEFAULT if usePretrainedFlag else None
        model = models.mobilenet_v3_small(weights=weights)
        head_attr = ("classifier", -1)
        default_img = 224
    elif arch in ("mbv3_large", "mobilenet_v3_large"):
        weights = MobileNet_V3_Large_Weights.DEFAULT if usePretrainedFlag else None
        model = models.mobilenet_v3_large(weights=weights)
        head_attr = ("classifier", -1)
        default_img = 224
    elif arch in ("effnet_b3", "efficientnet_b3"):
        weights = EfficientNet_B3_Weights.DEFAULT if usePretrainedFlag else None
        model = models.efficientnet_b3(weights=weights)
        head_attr = ("classifier", -1)
        default_img = 300
    elif arch in ("resnet50",):
        weights = ResNet50_Weights.DEFAULT if usePretrainedFlag else None
        model = models.resnet50(weights=weights)
        head_attr = ("fc", None)
        default_img = 224
    else:
        raise ValueError(f"Unknown arch: {arch}")

    # optionally freeze backbone
    if freeze:
        # Try to freeze all but the classification head
        for p in model.parameters():
            p.requires_grad_(False)

    # replace the head with our numClasses
    if head_attr[0] == "classifier":
        if isinstance(head_attr[1], int):
            inFeat = model.classifier[head_attr[1]].in_features
            model.classifier[head_attr[1]] = nn.Linear(inFeat, numClasses)
        else:
            # fallback: last layer
            inFeat = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(inFeat, numClasses)
    elif head_attr[0] == "fc":
        inFeat = model.fc.in_features
        model.fc = nn.Linear(inFeat, numClasses)

    return model, weights, default_img

def _ddp_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def runEpoch(model, loader, criterion, optimizer, device, scaler, trainMode: bool):
    model.train(trainMode)
    total, correct, lossSum = 0, 0, 0.0
    no_grad = torch.no_grad if not trainMode else torch.enable_grad
    with no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = model(imgs)
                loss = criterion(logits, labels)
            if trainMode:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            lossSum += loss.detach().item() * labels.size(0)
            preds = logits.argmax(1)
            correct += int((preds == labels).sum().item())
            total += labels.size(0)
    # Aggregate across processes if DDP
    if _ddp_ready():
        t = torch.tensor([lossSum, correct, total], dtype=torch.float64, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        lossSum, correct, total = t[0].item(), t[1].item(), t[2].item()
    return lossSum / max(1, total), correct / max(1, total)

def main():
    # DDP setup (single-process fallback otherwise)
    world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
    ddp = (world_size_env > 1) and torch.cuda.is_available()
    if ddp:
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        # make RNG different per-rank for better shuffling/augs
        setSeed(seed + dist.get_rank())
    else:
        setSeed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_rank = 0

    is_main = (not ddp) or (dist.get_rank() == 0)
    if is_main:
        print(f"[info] ddp={ddp} world_size={int(os.environ.get('WORLD_SIZE','1'))} local_rank={local_rank}")
        print(f"[info] device={device.type}, cuda={torch.cuda.is_available()}, workers={numWorkers}")
        print(f"[info] data={dataDir}")
        print(f"[info] models={saveDir}")

    # model + transforms
    model, weights, default_img = buildModel(
        numClasses=2, usePretrainedFlag=usePretrained, freeze=freezeBackbone, arch=arch
    )
    # override imgSize for some models
    img_size = int(os.environ.get("IMG_SIZE_OVERRIDE", default_img if imgSize == 224 else imgSize))
    trainTfms, evalTfms, mean, std = buildTransforms(weights, img_size)

    # data
    (trainSet, valSet, testSet), (trainLoader, valLoader, testLoader) = \
        buildDatasetsAndLoaders(dataDir, trainTfms, evalTfms, distributed=ddp)
    if is_main:
        print(f"[info] arch={arch}")
        print(f"[info] sizes: train={len(trainSet)}, val={len(valSet)}, test={len(testSet)}")
        print(f"[info] batches: train={len(trainLoader)}, val={len(valLoader)}, test={len(testLoader)}")

    # save metadata for inference
    meta = {
        "classes": trainSet.classes,
        "class_to_idx": trainSet.class_to_idx,
        "mean": [float(x) for x in mean],
        "std": [float(x) for x in std],
        "img_size": int(img_size),
        "arch": arch,
    }
    (saveDir / "meta.json").write_text(json.dumps(meta, indent=2))

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=baseLr, weight_decay=weightDecay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda',enabled=(device.type == "cuda"))  # AMP per PyTorch recipe

    # Wrap with DDP if enabled
    if ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[device.index], output_device=device.index)

    bestValAcc, bestPath = 0.0, saveDir / f"violence_{arch}_best.pt"

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        # ensure each epoch shuffles differently in DDP
        if ddp and hasattr(trainLoader, 'sampler') and hasattr(trainLoader.sampler, 'set_epoch'):
            trainLoader.sampler.set_epoch(epoch)
        trainLoss, trainAcc = runEpoch(model, trainLoader, criterion, optimizer, device, scaler, True)
        valLoss, valAcc = runEpoch(model, valLoader, criterion, optimizer, device, scaler, False)
        scheduler.step()
        dt = time.time() - t0
        if is_main:
            print(f"Epoch {epoch:02d}: "
                  f"trainLoss={trainLoss:.4f} acc={trainAcc:.3f} | "
                  f"valLoss={valLoss:.4f} acc={valAcc:.3f} | {dt:.1f}s")

        if is_main and valAcc > bestValAcc:
            bestValAcc = valAcc
            # handle DDP-wrapped models
            to_save = model.module.state_dict() if ddp else model.state_dict()
            torch.save(to_save, bestPath)
            print(f"  saved best -> {bestPath} (valAcc={bestValAcc:.3f})")

    # test with best
    if ddp:
        dist.barrier()
    if bestPath.exists():
        # load on all ranks so evaluation uses the same weights
        state = torch.load(bestPath, map_location=device)
        if ddp:
            model.module.load_state_dict(state)
        else:
            model.load_state_dict(state)
    testLoss, testAcc = runEpoch(model, testLoader, criterion, optimizer, device, scaler, False)
    if is_main:
        print(f"TEST: loss={testLoss:.4f} acc={testAcc:.3f}")

    # export ONNX (NCHW)
    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size, device=device)
    onnxPath = saveDir / "violence_classifier.onnx"
    if is_main:
        # export only once from rank-0
        torch.onnx.export(
            model.module if ddp else model, dummy, onnxPath.as_posix(),
            input_names=["input"], output_names=["logits"],
            opset_version=17, do_constant_folding=True
        )
        print(f"Exported ONNX -> {onnxPath}")

if __name__ == "__main__":
    main()
