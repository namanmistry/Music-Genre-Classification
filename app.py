from tensorflow import keras
import pandas as pd
import numpy as np
import scipy
import os
import pickle
import librosa
import librosa.display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import optimizers
from tensorflow.keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
pickle_open = open("converter.pkl","rb")
converter = pickle.load(pickle_open)
pickle_open.close()

pickle_open = open("fit.pkl","rb")
fit = pickle.load(pickle_open)
pickle_open.close()


integer_mapping = {l: i for i, l in enumerate(converter.classes_)}


def predict(song):
    songname = f"{song}"
    y, sr = librosa.load(songname, mono=True, duration=30)

    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)

    chroma_stft_mean = np.mean(chroma_stft)
    chroma_stft_var = np.var(chroma_stft)
    rms_mean = np.mean(rms)
    rms_var = np.var(rms)
    spec_cent_mean = np.mean(spec_cent)
    spec_cent_var = np.var(spec_cent)
    spec_bw_mean = np.mean(spec_bw)
    spec_bw_var = np.var(spec_bw)
    rolloff_mean = np.mean(rolloff)
    rolloff_var = np.var(rolloff)
    zcr_mean = np.mean(zcr)
    zcr_var= np.var(zcr)
    mfcc_list = []
    for i in range(0,20):
        mfcc_list.append(np.mean(mfcc[i]))
        mfcc_list.append(np.var(mfcc[i]))

    final_features = [chroma_stft_mean,chroma_stft_var,rms_mean,rms_var,spec_cent_mean,spec_cent_var,spec_bw_mean,spec_bw_var,rolloff_mean,rolloff_var,zcr_mean,zcr_var]+mfcc_list

    final_features = np.array(final_features)
    final_features = fit.transform([final_features])

    model = load_model("model.h5")

    return (converter.inverse_transform([np.argmax(model.predict(final_features.reshape(1,52)))]))[0]


app = Flask(__name__)
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/after",methods = ['POST', 'GET'])
def after():
    imagefile = request.files['myFile']
    imagefile.save(secure_filename(imagefile.filename))
    predicted_genre = predict(imagefile.filename)
    os.remove(imagefile.filename)
    return f"<h1>{predicted_genre}</h1>"

if __name__ == "__main__":
    app.run(debug=True)