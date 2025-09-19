# video_window_infer.py
import argparse, collections, json, time
from pathlib import Path

import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import (
    mobilenet_v3_small, mobilenet_v3_large,
    resnet50, efficientnet_b3, convnext_tiny,
    vit_b_16, swin_t
)

def buildModelForInfer(arch: str, numClasses: int):
    arch = arch.lower()
    if arch == "mobilenet_v3_small":
        m = mobilenet_v3_small(weights=None); inFeat = m.classifier[3].in_features; m.classifier[3] = nn.Linear(inFeat, numClasses)
    elif arch == "mobilenet_v3_large":
        m = mobilenet_v3_large(weights=None); inFeat = m.classifier[3].in_features; m.classifier[3] = nn.Linear(inFeat, numClasses)
    elif arch == "resnet50":
        m = resnet50(weights=None); inFeat = m.fc.in_features; m.fc = nn.Linear(inFeat, numClasses)
    elif arch == "efficientnet_b3":
        m = efficientnet_b3(weights=None); inFeat = m.classifier[1].in_features; m.classifier[1] = nn.Linear(inFeat, numClasses)
    elif arch == "convnext_tiny":
        m = convnext_tiny(weights=None); inFeat = m.classifier[2].in_features; m.classifier[2] = nn.Linear(inFeat, numClasses)
    elif arch == "vit_b_16":
        m = vit_b_16(weights=None); inFeat = m.heads.head.in_features; m.heads.head = nn.Linear(inFeat, numClasses)
    elif arch == "swin_t":
        m = swin_t(weights=None); inFeat = m.head.in_features; m.head = nn.Linear(inFeat, numClasses)
    else:
        raise ValueError(f"Unknown arch: {arch}")
    return m

def buildEvalTfm(mean, std, imgSize):
    return transforms.Compose([
        transforms.Resize(int(imgSize * 1.15)),
        transforms.CenterCrop(imgSize),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

@torch.no_grad()
def inferFrame(bgrFrame, tfm, model, classes, violenceIdx, device="cpu"):
    rgb = cv2.cvtColor(bgrFrame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    x = tfm(img).unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0].cpu().tolist()
    predIdx = int(torch.tensor(probs).argmax().item())
    return classes[predIdx], float(probs[violenceIdx])

def drawHud(frame, pred, pViolence, emaPv, thrStart, thrStop, alertOn, fps):
    h, w = frame.shape[:2]
    color = (0, 0, 255) if alertOn else (0, 200, 0)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)
    cv2.putText(frame, f"pred: {pred}", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"pV: {pViolence:.3f} | EMA: {emaPv:.3f} | thrStart/Stop: {thrStart:.2f}/{thrStop:.2f} | FPS: {fps:.1f}",
                (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
    cv2.rectangle(frame, (4,4), (w-4,h-4), color, 3)
    return frame

def computeMotionScore(prevGray, curGray):
    # simple frame-diff motion metric
    diff = cv2.absdiff(prevGray, curGray)
    score = float(diff.mean())
    return score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", type=str, default="models/effnetb3/meta.json")
    ap.add_argument("--model", type=str, default="models/effnetb3/violence_efficientnet_b3_best.pt")
    ap.add_argument("--source", type=str, default="0", help="camera index (e.g., 0/1) or video file path")
    ap.add_argument("--archOverride", type=str, default=None, help="force arch if meta lacks it")
    ap.add_argument("--window", type=int, default=30, help="window size (frames)")
    ap.add_argument("--minHits", type=int, default=10, help="min frames in window with p>=thrStart")
    ap.add_argument("--thrStart", type=float, default=0.8, help="start alert if EMA>=thrStart AND hits>=minHits")
    ap.add_argument("--thrStop", type=float, default=0.6, help="stop alert when EMA<thrStop")
    ap.add_argument("--emaAlpha", type=float, default=0.4, help="EMA smoothing factor (0..1)")
    ap.add_argument("--motionGate", type=float, default=4.0, help="skip/downweight frames if motion score < gate")
    ap.add_argument("--everyMs", type=int, default=150, help="infer every N ms")
    ap.add_argument("--saveAlertsDir", type=str, default=None)
    args = ap.parse_args()

    with open(args.meta, "r", encoding="utf-8") as f:
        meta = json.load(f)
    arch = args.archOverride or meta.get("arch", "efficientnet_b3")
    classes = meta["classes"]; mean, std = meta["mean"], meta["std"]; imgSize = meta["img_size"]
    if "violence" not in meta["class_to_idx"]:
        raise SystemExit(f'"violence" not in class_to_idx: {meta["class_to_idx"]}')
    violenceIdx = meta["class_to_idx"]["violence"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = buildModelForInfer(arch, numClasses=len(classes)).to(device)
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state); model.eval()
    tfm = buildEvalTfm(mean, std, imgSize)

    # open source
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    saveDir = Path(args.saveAlertsDir) if args.saveAlertsDir else None
    if saveDir: saveDir.mkdir(parents=True, exist_ok=True)

    cv2.namedWindow("Violence Detection (temporal)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Violence Detection (temporal)", 1024, 576)

    probsWin = collections.deque(maxlen=args.window)
    emaPv = 0.0
    alertOn = False
    lastEval = 0.0
    lastGray = None
    lastTs = time.time()
    fps = 0.0

    print("Press 'q' to quit, 's' to save a frame.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok: break

            # update FPS
            now = time.time()
            fps = 1.0 / (now - lastTs) if now > lastTs else fps
            lastTs = now

            # motion gating
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            motionScore = computeMotionScore(lastGray, gray) if lastGray is not None else args.motionGate
            lastGray = gray

            # throttle inference
            pViolence = 0.0
            pred = "n/a"
            msNow = now * 1000.0
            if msNow - lastEval >= args.everyMs:
                if motionScore >= args.motionGate:
                    pred, pViolence = inferFrame(frame, tfm, model, classes, violenceIdx, device)
                else:
                    # low motion → keep previous EMA gently decaying
                    pViolence = 0.5 * emaPv
                    pred = "non_violence"
                probsWin.append(pViolence)

                # EMA update
                alpha = args.emaAlpha
                emaPv = alpha * pViolence + (1 - alpha) * emaPv

                # hysteresis + majority rule
                hits = sum(1 for p in probsWin if p >= args.thrStart)
                if not alertOn and (emaPv >= args.thrStart and hits >= args.minHits):
                    alertOn = True
                elif alertOn and emaPv < args.thrStop:
                    alertOn = False

                lastEval = msNow

            shown = drawHud(frame.copy(), pred, pViolence, emaPv, args.thrStart, args.thrStop, alertOn, fps)
            cv2.imshow("Violence Detection (temporal)", shown)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            if key == ord('s'):
                out = saveDir if saveDir else Path("screens")
                out.mkdir(parents=True, exist_ok=True)
                ts = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(str(out / f"{ts}.jpg"), frame)
            if alertOn and saveDir:
                ts = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(str(saveDir / f"{ts}_ALERT.jpg"), frame)

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
