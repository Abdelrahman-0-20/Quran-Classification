# Quran Recitation Audio Classification

A machine learning project that classifies Quran recitations based on acoustic features extracted from audio files. The project includes a Streamlit web application for training a Random Forest classifier and performing predictions on new audio samples.

## Dataset

The dataset used for training and evaluation is available on Kaggle:

[Quran Recitations for Audio Classification](https://www.kaggle.com/datasets/mohammedalrajeh/quran-recitations-for-audio-classification?resource=download)

The dataset consists of audio recordings of Quran recitations organized in folders by reciter (class). To test the model with different reciters, download the dataset from the provided link and follow the training instructions below.

## Features

The following acoustic features are extracted from each audio file:

- **MFCC** (13 coefficients) – Mel-frequency cepstral coefficients capturing timbral texture.
- **Chroma** (12 coefficients) – Chroma energy normalized, representing pitch class distribution.
- **Spectral Contrast** (7 coefficients) – Difference in amplitude between peaks and valleys in the spectrum.

## Model

A Random Forest classifier is used for classification. The model is trained on the extracted features and saved as a pickle file (`model.pkl`) for later inference.

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/your-username/Quran-Classification.git
   cd Quran-Classification
   ```

2. Install the required Python packages:

   ```
   pip install -r requirements.txt
   ```

   If `requirements.txt` is not present, install the following manually:

   ```
   pip install streamlit pandas numpy librosa scikit-learn
   ```

## Usage

The application is built with Streamlit and provides two main pages accessible via the sidebar.

### Training the Model

1. Run the Streamlit app:

   ```
   streamlit run app.py
   ```

2. Navigate to the **Train Model** page using the sidebar.
3. Enter the full path to the dataset directory. The directory should contain subfolders named after each reciter (class), with `.wav` files inside them. For example:

   ```
   dataset/
   ├── reciter1/
   │   ├── audio1.wav
   │   ├── audio2.wav
   │   └── ...
   ├── reciter2/
   │   ├── audio1.wav
   │   └── ...
   └── ...
   ```

4. Click **Train Model**. The application will recursively scan for all `.wav` files, extract features, train a Random Forest classifier, evaluate it on a held-out test set, and save the model as `model.pkl` in the project directory.

### Making Predictions

1. Ensure a trained model (`model.pkl`) exists. If not, follow the training steps above.
2. In the Streamlit app, navigate to the **Predict** page.
3. Choose one of the following input methods:
   - **Upload WAV files**: Select one or multiple `.wav` files directly.
   - **Upload CSV with file paths**: Upload a CSV file containing a column named `path` (or any first column) with full paths to the audio files.
4. Click **Predict** to see the predicted class and confidence score for each file.

## Project Structure

```
Quran-Classification/
├── app.py                  # Main Streamlit application
├── model.pkl               # Trained model (generated after training)
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Notes

- The application assumes that the dataset directory uses subfolders to indicate class labels. If your dataset is structured differently, adjust the `train_model` function in `app.py` accordingly.
- The model is saved using `pickle`. If you prefer `joblib`, simply replace the save/load functions.
- For large datasets, feature extraction may take some time; a progress bar is displayed during training.

## Contributing

Contributions are welcome. Please open an issue or submit a pull request for any improvements.

## License

This project is provided without a specific license. Use it at your own discretion.
