#!/usr/bin/env python3.7

import argparse
import os
import numpy as np
import speech_recognition as sr
import whisper
import torch
import pyaudio

from datetime import datetime, timedelta
from queue import Queue
from time import sleep
from sys import platform


def getMicDeviceIndexByName(targetName: str) -> int:
    for idx, name in enumerate(sr.Microphone.list_microphone_names()):
        if targetName in name:
            return idx
    return -1


def getDefaultInputDeviceInfo():
    pa = pyaudio.PyAudio()
    try:
        info = pa.get_default_input_device_info()
    finally:
        pa.terminate()
    return info


def getDeviceInfoByIndex(index: int):
    pa = pyaudio.PyAudio()
    try:
        info = pa.get_device_info_by_index(index)
    finally:
        pa.terminate()
    return info


def resampleTo16k(x: np.ndarray, origSr: int) -> np.ndarray:
    if origSr == 16000:
        return x.astype(np.float32)
    newLen = int(round(len(x) * 16000.0 / float(origSr)))
    # linear interpolation resample (fast and good enough for ASR)
    return np.interp(
        np.linspace(0.0, 1.0, num=newLen, endpoint=False),
        np.linspace(0.0, 1.0, num=len(x), endpoint=False),
        x.astype(np.float32),
    ).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="large-v2", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--energy_threshold", default=1000, type=int,
                        help="Energy level for mic to detect.")
    parser.add_argument("--record_timeout", default=2.0, type=float,
                        help="How real time the recording is in seconds.")
    parser.add_argument("--phrase_timeout", default=3.0, type=float,
                        help="How much empty space between recordings before we consider it a new line.")
    if "linux" in platform:
        parser.add_argument("--default_microphone", default="RAMPAGE: USB Audio (hw:3,0)", type=str,
                            help="Default microphone name for SpeechRecognition. Use 'list' to show devices.")
    args = parser.parse_args()

    # recognizer config
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    recorder.dynamic_energy_threshold = False

    # choose microphone device + open at its native sample rate (avoid paInvalidSampleRate)
    deviceIndex = None
    micSampleRate = 16000  # fallback; will be overwritten below

    if "linux" in platform:
        micName = getattr(args, "default_microphone", None)
        if not micName or micName == "list":
            print("Available microphone devices:")
            for name in sr.Microphone.list_microphone_names():
                print(f'  - "{name}"')
            return
        deviceIndex = getMicDeviceIndexByName(micName)
        if deviceIndex < 0:
            print(f'Could not find a microphone containing "{micName}" in its name.')
            print("Use --default_microphone=list to see all devices.")
            return
        devInfo = getDeviceInfoByIndex(deviceIndex)
        micSampleRate = int(devInfo.get("defaultSampleRate", 48000))
        source = sr.Microphone(device_index=deviceIndex, sample_rate=micSampleRate)
    else:
        try:
            defaultInfo = getDefaultInputDeviceInfo()
            deviceIndex = int(defaultInfo.get("index", 0))
            micSampleRate = int(defaultInfo.get("defaultSampleRate", 48000))
            source = sr.Microphone(device_index=deviceIndex, sample_rate=micSampleRate)
        except Exception:
            # last resort: let SR pick defaults
            source = sr.Microphone()

    # load Whisper
    modelName = args.model
    audioModel = whisper.load_model(modelName)

    recordTimeout = float(args.record_timeout)
    phraseTimeout = float(args.phrase_timeout)

    # shared state
    dataQueue: Queue = Queue()
    transcription = [""]  # list of lines
    phraseBytes = bytes()
    lastPhraseTime = None

    # callback collects raw bytes
    def recordCallback(_, audio: sr.AudioData) -> None:
        dataQueue.put(audio.get_raw_data())

    # open stream and set ambient noise
    try:
        with source:
            recorder.adjust_for_ambient_noise(source, duration=0.5)
    except OSError as e:
        print(f"[Mic open error] {e}")
        return

    # start background listener
    stopListening = recorder.listen_in_background(source, recordCallback, phrase_time_limit=recordTimeout)
    print(f"Microphone ready @ {micSampleRate} Hz | Whisper model: {modelName}")
    print("Speak into the mic. Press Ctrl+C to stop.\n")

    try:
        while True:
            try:
                now = datetime.utcnow()
                if not dataQueue.empty():
                    phraseComplete = False
                    if lastPhraseTime and now - lastPhraseTime > timedelta(seconds=phraseTimeout):
                        phraseBytes = bytes()
                        phraseComplete = True
                    lastPhraseTime = now

                    # drain queue
                    chunkList = []
                    while not dataQueue.empty():
                        chunkList.append(dataQueue.get())
                    if chunkList:
                        phraseBytes += b"".join(chunkList)

                    # convert to float32 waveform in [-1, 1]
                    audioNp = np.frombuffer(phraseBytes, dtype=np.int16).astype(np.float32) / 32768.0
                    # resample to 16k for Whisper
                    audioNp = resampleTo16k(audioNp, micSampleRate)

                    # transcribe
                    result = audioModel.transcribe(audioNp, fp16=torch.cuda.is_available())
                    text = result.get("text", "").strip()

                    if phraseComplete:
                        transcription.append(text)
                    else:
                        transcription[-1] = text

                    # redraw
                    os.system("cls" if os.name == "nt" else "clear")
                    for line in transcription:
                        print(line)
                    print("", end="", flush=True)
                else:
                    sleep(0.25)
            except KeyboardInterrupt:
                break
    finally:
        try:
            stopListening(wait_for_stop=False)
        except Exception:
            pass

    print("\n\nTranscription:")
    for line in transcription:
        print(line)


if __name__ == "__main__":
    main()
