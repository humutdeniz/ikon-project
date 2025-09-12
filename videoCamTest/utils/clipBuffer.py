import collections, numpy as np, cv2

class ClipBuffer:
    def __init__(self, clipSeconds: float, sampleFps: int):
        self.clipSeconds = clipSeconds
        self.sampleFps = sampleFps
        self.maxFrames = int(round(clipSeconds * sampleFps))
        self.frames = collections.deque(maxlen=self.maxFrames)
        self.timestamps = collections.deque(maxlen=self.maxFrames)
        self._lastPushTs = None  # unused; kept for potential future time-based sampling

    def push(self, frame, ts):
        # push every frame; downsample uniformly in getClip
        self.frames.append(frame)
        self.timestamps.append(ts)

    def hasEnough(self):
        return len(self.frames) >= self.maxFrames

    def getClip(self, targetSize: int):
        if not self.hasEnough():
            return None, None
        # uniform sample to exactly maxFrames
        idx = np.linspace(0, len(self.frames)-1, self.maxFrames).astype(np.int32)
        imgs = [self.frames[i] for i in idx]
        # resize & center-crop to square targetSize
        proc = [self._resizeSquare(f, targetSize) for f in imgs]
        return proc, list(self.timestamps)

    def _resizeSquare(self, img, size):
        h, w = img.shape[:2]
        scale = size / min(h, w)
        nh, nw = int(round(h*scale)), int(round(w*scale))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        # center crop
        y0 = max((nh - size)//2, 0)
        x0 = max((nw - size)//2, 0)
        return resized[y0:y0+size, x0:x0+size]
