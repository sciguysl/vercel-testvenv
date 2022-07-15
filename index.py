from flask import Flask
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model


with open("x_SCN", "rb") as f:
    X = pickle.load(f)

with open("y_SCN", "rb") as f:
    y = pickle.load(f)


labelencoder=LabelEncoder()
model = load_model('model_SCN.h5')

y=to_categorical(labelencoder.fit_transform(y))


#@title Default title text
from os import walk
import librosa
f = []
mypath9="./input"
x=[mypath9]

for i in x:
    for (dirpath, dirnames, filenames) in walk(i):
        f.extend(filenames)
        break
# print(f)

d={}
for i in f:
    if 'SCN' in i:
        path = "./input"
        file = path + '/' + i
        d[i] = file
    if 'SVA' in i:
        path = "./input"
        file = path + '/' + i
        d[i] = file
    elif 'SVE' in i:
        path = "./input"
        file = path + '/' + i
        d[i] = file
    elif 'SVO' in i:
        path = "./input"
        file = path + '/' + i
        d[i] = file
    if 'BD' in i:
        path = "./input"
        file = path + '/' + i
        d[i] = file
    elif 'BS' in i:
        path ="./input"
        file = path + '/' + i
        d[i] = file

df=pd.DataFrame.from_dict(d,orient='index',columns=['relative_path'])


app = Flask(__name__)


prediction = ''


@app.route('/')
def home():
    for index_num,row in tqdm(df.iterrows()):
	    file_name = str(row["relative_path"])
	    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
	    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=80)
	    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
	    #Reshape MFCC feature to 2-D array
	    mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
	    #predicted_label=model.predict_classes(mfccs_scaled_features)
	    x_predict=model.predict(mfccs_scaled_features)
	    predicted_label=np.argmax(x_predict,axis=1)
	    prediction_class = labelencoder.inverse_transform(predicted_label)
	    prediction = str(prediction_class)
    return '<h1>' + prediction + '</h1>'



@app.route('/about')
def about():
    return 'About Page Route'


@app.route('/portfolio')
def portfolio():
    return 'Portfolio Page Route'


@app.route('/contact')
def contact():
    return 'Contact Page Route'


@app.route('/api')
def api():
    with open('data.json', mode='r') as my_file:
        text = my_file.read()
        return text
