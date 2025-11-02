import argparse
import pickle
import numpy as np

from audio_features import extract_features


def main():
    ap = argparse.ArgumentParser(description="Predict mood for an audio file using a trained model.")
    ap.add_argument("--model", required=True, help="Path to model.pkl saved by train.py")
    ap.add_argument("--file", required=True, help="Path to audio file (wav recommended if librosa not available).")
    args = ap.parse_args()

    with open(args.model, "rb") as f:
        model = pickle.load(f)

    x, feat_names = extract_features(args.file)

    model_feats = model["feature_names"]
    if feat_names != model_feats:
        raise RuntimeError("Feature schema mismatch between model and input audio.")

    X = x.reshape(1, -1)

    if model.get("type") == "sklearn":
        clf = model["model"]
        pred = clf.predict(X).tolist()[0]
        print(f"[v0] Predicted mood: {pred}")
        try:
            proba = getattr(clf, "predict_proba", None)
            if callable(proba):
                p = proba(X)[0]
                labels = clf.classes_
                idx = np.argsort(p)[::-1][:3]
                print("[v0] Top probabilities:")
                for i in idx:
                    print(f"  {labels[i]}: {p[i]:.3f}")
        except Exception:
            pass
    else:
        mean = model["mean"]
        std = model["std"]
        Xn = (X - mean) / std
        Xt = model["X"]
        k = model["k"]
        y = model["y"]
        d = np.sqrt(((Xt - Xn[0]) ** 2).sum(axis=1))
        nn_idx = np.argsort(d)[:k]
        nn_labels = [y[j] for j in nn_idx]
        vals, counts = np.unique(nn_labels, return_counts=True)
        pred = vals[np.argmax(counts)]
        print(f"[v0] Predicted mood: {pred}")
        print("[v0] Nearest neighbors:")
        for j in nn_idx[:5]:
            print(f"  neighbor label={y[j]} dist={d[j]:.4f}")


if __name__ == "__main__":
    main()
