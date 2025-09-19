# file: piperFixWav.py
import io, os, wave
from pathlib import Path
from piper.voice import PiperVoice

def writeWav(text: str, modelPath: str, outPath: str):
    modelAbs = str(Path(modelPath).resolve())
    cfgAbs = modelAbs.replace(".onnx", ".onnx.json")
    if not os.path.isfile(modelAbs) or not os.path.isfile(cfgAbs):
        raise FileNotFoundError("Model veya JSON bulunamadı")

    voice = PiperVoice.load(modelAbs)
    sampleRate = int(voice.config.sample_rate) or 22050

    mem = io.BytesIO()
    with wave.open(mem, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sampleRate)
        voice.synthesize(text, w)

    mem.seek(0)
    with wave.open(mem, "rb") as r:
        nFrames = r.getnframes()
        sr = r.getframerate()
        print(f"[info] frames={nFrames} sr={sr} dur={nFrames/float(sr) if sr else 0:.3f}s")
        pcm = r.readframes(nFrames)

    with wave.open(outPath, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(sr)
        out.writeframes(pcm)
    print(f"[ok] wrote {outPath}")

if __name__ == "__main__":
    writeWav(
        text="Merhaba! Piper Türkçe konuşuyor. İkinci cümleyi de ekliyorum.",
        modelPath="src/voices/tr_TR-dfki-medium.onnx",
        outPath="sample.wav",
    )
