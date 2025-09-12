
import os, cv2, yaml, time, argparse
import torch
from utils.common import setDeterministic, ensureDir
from utils.videoReader import VideoReader
from utils.clipBuffer import ClipBuffer
from utils.hysteresis import HysteresisDecider
from utils.personGate import PersonGate
from utils.logger import Logger
from models.stageAX3d import StageAX3d
from models.stageBSlowfast import StageBSlowfast

def parseArgs():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/defaultConfig.yaml")
    return p.parse_args()

def main():
    args = parseArgs()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    cfgDir = os.path.dirname(args.config)
    # resolve relative checkpoint paths against the config directory
    for k in ("stageA", "stageB"):
        if cfg.get(k, {}).get("checkpointPath"):
            p = cfg[k]["checkpointPath"]
            if not os.path.isabs(p):
                rp = os.path.join(cfgDir, p)
                if os.path.exists(rp):
                    cfg[k]["checkpointPath"] = rp

    setDeterministic(42)
    # resolve logging output dir
    outDir = cfg["logging"]["outputDir"]
    if not os.path.isabs(outDir):
        outDir = os.path.join(cfgDir, outDir)
    ensureDir(outDir)
    cfg["logging"]["outputDir"] = outDir
    logger = Logger(cfg["logging"]["outputDir"], cfg["logging"]["level"])

    # Models
    aNorm = cfg.get("stageA", {}).get("normalize")
    aMean = cfg.get("stageA", {}).get("mean")
    aStd  = cfg.get("stageA", {}).get("std")
    aSkip = bool(cfg.get("stageA", {}).get("skipNorm", False))
    stageA = StageAX3d(device=cfg["stageA"]["device"], checkpointPath=cfg["stageA"]["checkpointPath"],
                       normalize=aNorm, mean=aMean, std=aStd, skipNorm=aSkip) if cfg["stageA"]["enabled"] else None
    stageB = StageBSlowfast(device=cfg["stageB"]["device"], checkpointPath=cfg["stageB"]["checkpointPath"]) if cfg["stageB"]["enabled"] else None

    # IO
    reader = VideoReader(cfg["videoSource"])
    buffer = ClipBuffer(cfg["clipSeconds"], cfg["sampleFps"])
    decider = HysteresisDecider(cfg["decision"]["alertOn"], cfg["decision"]["clearOn"],
                                cfg["decision"]["minRaiseSeconds"], cfg["decision"]["minClearSeconds"])
    gate = PersonGate(cfg["personGate"]["enabled"], cfg["personGate"]["minPersonArea"])
    lastEval = 0.0
    stride = cfg["strideSeconds"]
    useDisplay = bool(cfg["useDisplay"])
    displayScale = float(cfg["displayScale"])
    inputSize = int(cfg["inputSize"])
    dbg = cfg.get("debug", {})
    showClipGrid = bool(dbg.get("showClipGrid", False))

    logger.info("pipelineStarted", {"devices": torch.cuda.device_count()})
    alerting = False
    lastTxt = ""

    prevGray = None
    try:
        while True:
            ok, frame, ts = reader.read()
            if not ok:
                logger.warn("frameReadFailed")
                time.sleep(0.02)
                continue

            buffer.push(frame, ts)

            # simple motion metric (mean abs diff) to detect a static feed
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            motion = 0.0
            if prevGray is not None and prevGray.shape == gray.shape:
                motion = float(cv2.absdiff(gray, prevGray).mean())
            prevGray = gray

            if useDisplay:
                disp = cv2.resize(frame, None, fx=displayScale, fy=displayScale)
                status = "ALERT" if alerting else "SAFE"
                cv2.putText(disp, status, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255) if alerting else (0,255,0), 2)
                if lastTxt:
                    color = (0,0,255) if alerting else (0,255,0)
                    cv2.putText(disp, lastTxt, (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(disp, f"motion={motion:.1f}", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
                cv2.imshow("violenceGuard", disp)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):
                    break

            # evaluate on stride
            if (ts - lastEval) < stride:
                continue
            lastEval = ts

            if not buffer.hasEnough():
                continue

            if not gate.allow(frame):
                logger.debug("personGateBlocked")
                continue

            clip, _ = buffer.getClip(inputSize)
            if clip is None:
                continue

            # Stage A
            probA = stageA.forwardClip(clip) if stageA else 0.0

            # If high or periodic verify, Stage B
            doVerify = (probA >= cfg["stageA"]["threshold"]) or True  # also periodic verify; keep True for simplicity
            if stageB and doVerify:
                probB = stageB.forwardClip(clip)
                prob = max(probA, probB)  # conservative (raise on stronger evidence)
            else:
                prob = probA
            
            txt = f"pA={probA:.2f} pB={probB:.2f}" if stageB else f"pA={probA:.2f}"
            lastTxt = txt

            alerting, state = decider.update(prob)
            logger.info("eval", {"probA": round(probA,4), "probB": round(probB,4) if stageB else None, "prob": round(prob,4), "alert": alerting, "state": state, "motion": round(motion,2)})

            # TODO: if cfg.logging.saveAlertsMp4, dump a rolling mp4 around alert times

            # optional: visualize the actual clip being fed (grid of frames)
            if useDisplay and showClipGrid:
                try:
                    import numpy as np
                    T = len(clip)
                    cols = int(np.ceil(np.sqrt(T)))
                    rows = int(np.ceil(T / cols))
                    h, w = clip[0].shape[:2]
                    grid = np.zeros((rows*h, cols*w, 3), dtype=clip[0].dtype)
                    for i, fr in enumerate(clip):
                        r, c = divmod(i, cols)
                        grid[r*h:(r+1)*h, c*w:(c+1)*w] = fr
                    grid = cv2.resize(grid, None, fx=displayScale*0.6, fy=displayScale*0.6)
                    cv2.imshow("clipGrid", grid)
                except Exception:
                    pass

    except KeyboardInterrupt:
        logger.info("shutdownRequested")
    finally:
        reader.release()
        if useDisplay:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
