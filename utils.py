# utils.py - small helpers (placeholder for dataset loading)
import numpy as np
EMOTION_LABELS = ['angry','disgust','fear','happy','sad','surprise','neutral']

def decode_emotion(onehot):
    idx = int(onehot.argmax())
    return EMOTION_LABELS[idx]
