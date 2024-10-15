import os
import librosa
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications.vgg19 import preprocess_input

app = Flask(__name__)

# Load the model once at startup
model = None

# Function to extract MFCC features
def extract_mfcc(file_path, n_mfcc=13):
    audio, sample_rate = librosa.load(file_path, sr=None)  # Load with original sampling rate
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    mfcc_scaled = np.mean(mfcc.T, axis=0)
    return mfcc_scaled

# Preprocess the MFCC features for VGG19
def preprocess_mfcc_for_vgg(mfcc_scaled):
    mfcc_reshaped = np.zeros((224, 224))
    mfcc_reshaped[:mfcc_scaled.shape[0], :mfcc_scaled.shape[0]] = mfcc_scaled[:, np.newaxis]
    mfcc_reshaped = np.stack((mfcc_reshaped,) * 3, axis=-1)
    mfcc_reshaped = preprocess_input(mfcc_reshaped)
    return mfcc_reshaped

# Create or load the VGG19 model for prediction
def create_vgg19_model():
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers:
        layer.trainable = False
    x = Flatten()(base_model.output)
    x = Dense(256, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=x)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            # Save the uploaded file
            audio_file_path = os.path.join('uploads', file.filename)
            file.save(audio_file_path)

            # Extract features and make prediction
            mfcc_scaled = extract_mfcc(audio_file_path)
            mfcc_preprocessed = preprocess_mfcc_for_vgg(mfcc_scaled)
            mfcc_preprocessed = np.expand_dims(mfcc_preprocessed, axis=0)

            # Make prediction
            prediction = model.predict(mfcc_preprocessed)

            # Adjust the threshold if necessary (e.g., 0.6 for fake calls)
            threshold = 0.5
            result = "Fake 911 call" if prediction[0] > threshold else "Real 911 call"
            return render_template('index.html', result=result)

    return render_template('index.html', result=None)

if __name__ == '__main__':
    # Load the model
    model = create_vgg19_model()
    # Uncomment the following line if you have a saved model
    # model.load_weights('path_to_your_model_weights.h5')
    app.run(debug=True)
