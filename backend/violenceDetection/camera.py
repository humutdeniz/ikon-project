# 5_realtime_infer.py
import cv2, time, numpy as np, tensorflow as tf
from collections import deque

model = tf.saved_model.load("export/saved_model")
infer = model.signatures["serving_default"]  # inputs: image -> logits
inputKey = list(infer.structured_input_signature[1].keys())[0]
outputKey = list(infer.structured_outputs.keys())[0]

numFrames = 32
size = 224
stride = 2  # sample every 2 frames
threshold = 0.5

frameBuf = deque(maxlen=numFrames)

cap = cv2.VideoCapture(0)  # or path/RTSP
frameCount = 0

def preprocessFrame(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    x = rgb.astype(np.float32) / 255.0
    return x

def predictClip(clip):
    # clip: (T,H,W,3) float32 in [0,1]
    clip = np.expand_dims(clip, axis=0)  # (1,T,H,W,3)
    out = infer(**{inputKey: tf.convert_to_tensor(clip)})[outputKey].numpy()  # (1,2) logits
    probs = tf.nn.softmax(out, axis=-1).numpy()[0]
    return float(probs[1])  # index 1 = fight

while True:
    ok, frame = cap.read()
    if not ok: break
    if frameCount % stride == 0:
        frameBuf.append(preprocessFrame(frame))
    frameCount += 1

    pFight = 0.0
    if len(frameBuf) == numFrames:
        clip = np.stack(list(frameBuf), axis=0)
        pFight = predictClip(clip)

    label = f"FIGHT {pFight:.2f}" if pFight >= threshold else f"NO-FIGHT {(1-pFight):.2f}"
    color = (0,0,255) if pFight >= threshold else (0,255,0)
    cv2.putText(frame, label, (16,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.imshow("Violence Detection (MoViNet-A0)", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Esc
        break

cap.release()
cv2.destroyAllWindows()
