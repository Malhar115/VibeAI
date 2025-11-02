AI-Based Music Mood Classification — How to Run (in v0)

What you have:
- scripts/generate_synthetic_dataset.py — makes a small dataset with two moods: calm and energetic
- scripts/train.py — extracts features and trains a classifier (LogisticRegression if scikit-learn is available, otherwise a built-in KNN)
- scripts/predict.py — predicts the mood for a given audio file
- scripts/audio_features.py — feature extraction: ZCR, energy, spectral centroid/bandwidth/rolloff/flatness; optional MFCC/chroma/contrast if librosa is available
- scripts/utils.py — helpers for data listing and splitting

Important notes:
- These scripts run without external dependencies. If librosa/scikit-learn are present, they are used automatically. Otherwise, the code falls back gracefully.
- Without librosa, only .wav files are supported.
- Keep your dataset as: data/<dataset-name>/<label>/*.wav (e.g., data/myset/happy/*.wav)

Step-by-step (no terminal needed):
1) Generate a toy dataset
- Open scripts/generate_synthetic_dataset.py and click Run.
- This creates data/synth with calm/ and energetic/ WAV files (5s clips).
- You can change arguments at the top by editing defaults (outdir, n-per-class, duration, sr) if needed.

2) Train the model
- Open scripts/train.py and click Run.
- Default args expect: --data-dir data/synth  (You can edit the default line: ap.add_argument to change directory, test size, etc.)
- The script prints evaluation metrics and saves model.pkl at the project root (or the path you set with --output).

3) Predict on a file
- Open scripts/predict.py and click Run.
- By default, set the arguments at the top via the ap.add_argument defaults or edit before running:
  --model model.pkl
  --file data/synth/energetic/energetic_01.wav
- The script prints the predicted mood and (if sklearn) class probabilities; for KNN, it prints nearest neighbors.

Using your own audio:
- Put files into data/your-dataset/<mood>/*.wav (at least a few samples per mood).
- Edit scripts/train.py default argument for --data-dir to data/your-dataset and Run again.
- Use scripts/predict.py with --file path/to/your_file.wav and the saved --model.

Tips:
- Feature consistency is enforced: the model stores feature_names and predictions verify the same order/schema.
- You can switch to the simple KNN even if sklearn is available by adding --force-fallback in train.py (set it True by default or pass it).
- For best results in small datasets, try increasing --n-per-class in the synthetic generator and/or lowering --test-size in training.

Directory scaffold you can follow for custom data:
data/
  myset/
    happy/
      track1.wav
      track2.wav
    sad/
      song1.wav
      song2.wav
    energetic/
      file1.wav
