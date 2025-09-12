
import cv2, time

class VideoReader:
    def __init__(self, source, width=None, height=None):
        self.cap = cv2.VideoCapture(source)
        if width is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self):
        ok, frame = self.cap.read()
        ts = time.time()
        return ok, frame, ts

    def release(self):
        self.cap.release()
