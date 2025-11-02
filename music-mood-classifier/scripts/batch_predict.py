import os
import pickle
from audio_features import extract_features
from utils import list_audio_files_with_labels

def main():
    model_path = "model.pkl"
    data_dir = "data/synth"
    
    # Load model
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    # Get all audio files
    pairs = list_audio_files_with_labels(data_dir)
    
    print(f"Predicting on {len(pairs)} files:")
    print("-" * 50)
    
    correct = 0
    for file_path, true_label in pairs:
        # Extract features
        x, feat_names = extract_features(file_path)
        
        # Predict
        if model["type"] == "sklearn":
            probs = model["model"].predict_proba([x])[0]
            pred_label = model["model"].predict([x])[0]
            labels = model["labels"]
            prob_dict = {labels[i]: probs[i] for i in range(len(labels))}
        else:
            # KNN fallback
            from train import knn_predict
            pred_label = knn_predict(model, x.reshape(1, -1))[0]
            prob_dict = {pred_label: 1.0}
        
        # Check accuracy
        is_correct = pred_label == true_label
        if is_correct:
            correct += 1
        
        filename = os.path.basename(file_path)
        status = "OK" if is_correct else "FAIL"
        print(f"{status} {filename:20} | True: {true_label:10} | Pred: {pred_label:10} | Conf: {max(prob_dict.values()):.3f}")
    
    print("-" * 50)
    print(f"Accuracy: {correct}/{len(pairs)} ({100*correct/len(pairs):.1f}%)")

if __name__ == "__main__":
    main()
