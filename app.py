import streamlit as st
import pandas as pd
import numpy as np
import librosa
import pickle
import os
import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Audio Classifier", layout="centered")

MODEL_PATH = "model.pkl"


# Feature extraction
 
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

    feature_dict = {}
    for i, val in enumerate(mfccs.mean(axis=1)):
        feature_dict[f"MFCC_{i+1}"] = val
    for i, val in enumerate(chroma.mean(axis=1)):
        feature_dict[f"Chroma_{i+1}"] = val
    for i, val in enumerate(spectral_contrast.mean(axis=1)):
        feature_dict[f"Spectral_Contrast_{i+1}"] = val

    return pd.Series(feature_dict)

# 
# Model persistence
 
def save_model(model):
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return None

 
# Training
# 
def train_model(dataset_dir):
    # Collect all WAV files recursively
    wav_files = glob.glob(os.path.join(dataset_dir, "**", "*.wav"), recursive=True)
    if not wav_files:
        st.error("No WAV files found in the given directory.")
        return None

    features_list = []
    labels = []
    errors = []

    progress = st.progress(0)
    total = len(wav_files)

    for idx, file_path in enumerate(wav_files):
        # Label is the name of the parent folder
        label = os.path.basename(os.path.dirname(file_path))
        try:
            features = extract_features(file_path)
            features_list.append(features)
            labels.append(label)
        except Exception as e:
            errors.append(file_path)
        progress.progress((idx + 1) / total)

    if errors:
        st.warning(f"Skipped {len(errors)} files due to errors.")

    if not features_list:
        st.error("No valid audio files processed.")
        return None

    X = pd.DataFrame(features_list)
    y = np.array(labels)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.success(f"Model trained on {len(X)} samples. Test accuracy: {acc:.2f}")

    save_model(model)
    return model

 
# UI
# 
st.title("Audio Classification")

page = st.sidebar.radio("Navigate", ["Train Model", "Predict"])

if page == "Train Model":
    st.header("Train Model from Dataset Directory")
    st.write("Enter the path to the dataset directory. The directory should contain subfolders named after each class, with .wav files inside them.")
    dataset_dir = st.text_input("Dataset directory path", value="")
    
    if st.button("Train Model"):
        if dataset_dir and os.path.isdir(dataset_dir):
            with st.spinner("Training..."):
                model = train_model(dataset_dir)
            if model:
                st.success("Model saved as model.pkl")
        else:
            st.error("Please enter a valid directory path.")

elif page == "Predict":
    st.header("Predict Audio Classes")
    model = load_model()
    if model is None:
        st.warning("No trained model found. Please go to Train Model page first.")
        st.stop()

    # Choose input method
    input_method = st.radio("Select input method:", ["Upload WAV files", "Upload CSV with file paths"])

    file_paths = []
    file_names = []

    if input_method == "Upload WAV files":
        uploaded_files = st.file_uploader("Choose WAV file(s)", type=['wav'], accept_multiple_files=True)
        if uploaded_files:
            file_paths = uploaded_files
            file_names = [f.name for f in uploaded_files]
    else:
        csv_file = st.file_uploader("Upload CSV file (must contain a 'path' column)", type=['csv'])
        if csv_file is not None:
            try:
                df = pd.read_csv(csv_file)
                if 'path' in df.columns:
                    path_col = 'path'
                else:
                    path_col = df.columns[0]
                paths = df[path_col].astype(str).tolist()
                for p in paths:
                    if os.path.exists(p):
                        file_paths.append(p)
                        file_names.append(os.path.basename(p))
                    else:
                        st.warning(f"File not found: {p}")
            except Exception as e:
                st.error(f"Error reading CSV: {e}")

    if file_paths:
        if st.button("Predict"):
            results = []
            all_features = []

            for path, name in zip(file_paths, file_names):
                st.markdown(f"**File:** {name}")
                if hasattr(path, 'read'):
                    st.audio(path, format='audio/wav')
                else:
                    st.audio(path)

                try:
                    features = extract_features(path)
                    all_features.append(features)
                    features_reshaped = features.values.reshape(1, -1)
                    prediction = model.predict(features_reshaped)[0]
                    try:
                        proba = model.predict_proba(features_reshaped)[0]
                        confidence = f"{max(proba)*100:.1f}%"
                    except AttributeError:
                        confidence = "N/A"

                    results.append({"File": name, "Prediction": prediction, "Confidence": confidence})
                    st.write(f"**Prediction:** {prediction}  (Confidence: {confidence})")
                except Exception as e:
                    st.error(f"Error processing {name}: {e}")
                st.markdown("---")

            if results:
                if len(results) > 1:
                    st.subheader("All Results")
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df)

                if all_features:
                    st.subheader("Extracted Features (first file)")
                    feature_df = pd.DataFrame(all_features[0]).T
                    st.dataframe(feature_df)
    else:
        st.info("Please provide WAV files or a CSV with file paths.")
