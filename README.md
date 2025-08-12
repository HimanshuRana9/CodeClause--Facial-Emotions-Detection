# Facial Emotions Detection (Project ID: #CC3604)

**Quick start (demo-ready)**

1. Create a Python 3.8+ virtual environment and activate it.
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Run the live demo (uses included sample model `emotion_model.h5`):
   ```bash
   python realtime.py
   ```
3. Press `q` to quit the webcam window.

**Notes**
- The included model is a small demo model to verify the pipeline quickly. For real training, add the FER2013 dataset and run `train.py`.
- To retrain, place `fer2013.csv` under `data/` or prepare a directory structure of images by emotion, then run `python train.py`.
