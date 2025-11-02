import argparse
import os
import pickle
from typing import List, Tuple, Dict

import numpy as np

from audio_features import extract_features
from utils import list_audio_files_with_labels, stratified_split

# Optional sklearn
HAS_SK = False
try:
    from sklearn.pipeline import Pipeline  # type: ignore
    from sklearn.preprocessing import StandardScaler  # type: ignore
    from sklearn.linear_model import LogisticRegression  # type: ignore
    from sklearn.metrics import classification_report, confusion_matrix  # type: ignore
    HAS_SK = True
except Exception:
    HAS_SK = False


def compute_features(pairs: List[Tuple[str, str]]) -> Tuple[np.ndarray, List[str], List[str]]:
    X_list: List[np.ndarray] = []
    y_list: List[str] = []
    names_ref: List[str] = []
    for i, (path, label) in enumerate(pairs):
        vec, names = extract_features(path)
        if i == 0:
            names_ref = names
        else:
            if names != names_ref:
                raise RuntimeError(f"Feature schema mismatch for {path}")
        X_list.append(vec)
        y_list.append(label)
        print(f"[v0] Extracted {vec.shape[0]} features from {os.path.basename(path)} -> label={label}")
    X = np.vstack(X_list).astype(np.float32)
    return X, y_list, names_ref


def train_sklearn(X_train, y_train):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    pipe.fit(X_train, y_train)
    return pipe


def evaluate_basic(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    acc = (np.array(y_true) == np.array(y_pred)).mean().item()
    return {"accuracy": float(acc)}


def knn_fit(X: np.ndarray, y: List[str], k: int = 5) -> Dict:
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    Xn = (X - mean) / std
    return {"type": "knn", "X": Xn.astype(np.float32), "y": y, "k": int(k), "mean": mean.astype(np.float32), "std": std.astype(np.float32)}


def knn_predict(model: Dict, X: np.ndarray) -> List[str]:
    mean = model["mean"]
    std = model["std"]
    Xn = (X - mean) / std
    Xt = model["X"]
    k = model["k"]
    y = model["y"]
    preds: List[str] = []
    for i in range(Xn.shape[0]):
        d = np.sqrt(((Xt - Xn[i]) ** 2).sum(axis=1))
        nn_idx = np.argsort(d)[:k]
        nn_labels = [y[j] for j in nn_idx]
        vals, counts = np.unique(nn_labels, return_counts=True)
        pred = vals[np.argmax(counts)]
        preds.append(pred)
    return preds


def main():
    ap = argparse.ArgumentParser(description="Train a music mood classifier from labeled audio folders.")
    ap.add_argument("--data-dir", required=True, help="Root folder with subfolders per label (e.g., data/happy, data/sad).")
    ap.add_argument("--output", default="model.pkl", help="Output model file (.pkl).")
    ap.add_argument("--test-size", type=float, default=0.2, help="Fraction for test split.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=5, help="k for KNN fallback (if sklearn unavailable).")
    ap.add_argument("--force-fallback", action="store_true", help="Use custom KNN even if sklearn is available.")
    args = ap.parse_args()

    pairs = list_audio_files_with_labels(args.data_dir)
    if len(pairs) == 0:
        print("[v0] No audio files found. Ensure structure: <data-dir>/<label>/*.wav")
        return

    print(f"[v0] Found {len(pairs)} files across {len(set([l for _, l in pairs]))} labels.")
    train_pairs, test_pairs = stratified_split(pairs, test_size=args.test_size, seed=args.seed)

    print(f"[v0] Train: {len(train_pairs)} | Test: {len(test_pairs)}")
    X_train, y_train, feat_names = compute_features(train_pairs)
    X_test, y_test, feat_names2 = compute_features(test_pairs)
    assert feat_names == feat_names2

    model_blob: Dict[str, object] = {
        "feature_names": feat_names,
        "labels": sorted(set(y_train + y_test)),
    }

    use_sklearn = (HAS_SK and not args.force_fallback)
    if use_sklearn:
        print("[v0] Training scikit-learn LogisticRegression classifier...")
        clf = train_sklearn(X_train, y_train)
        y_pred = clf.predict(X_test).tolist()
        print("[v0] Evaluation:")
        try:
            print(classification_report(y_test, y_pred))
            print("Confusion matrix:")
            print(confusion_matrix(y_test, y_pred))
        except Exception:
            metrics = evaluate_basic(y_test, y_pred)
            print(metrics)
        model_blob.update({"type": "sklearn", "model": clf})
    else:
        print("[v0] Training fallback KNN classifier...")
        knn = knn_fit(X_train, y_train, k=args.k)
        y_pred = knn_predict(knn, X_test)
        metrics = evaluate_basic(y_test, y_pred)
        print("[v0] Evaluation:", metrics)
        model_blob.update(knn)

    with open(args.output, "wb") as f:
        pickle.dump(model_blob, f)
    print(f"[v0] Saved model -> {args.output}")
    print(f"[v0] Feature count: {len(feat_names)} | Labels: {model_blob['labels']}")


if __name__ == "__main__":
    main()
