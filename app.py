from flask import Flask, render_template, request, flash, redirect, url_for
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
from werkzeug.utils import secure_filename
import librosa
import torch
import os

tokenizer = Wav2Vec2Tokenizer.from_pretrained(
    "wav2vec2-large-xlsr-arabic")
model = Wav2Vec2ForCTC.from_pretrained("wav2vec2-large-xlsr-arabic")
audio_file = 'upload/audio.wav'

app = Flask(__name__)
app.secret_key = "AymenMohammed"

def prepare_example(example):
    speech, sampling_rate = librosa.load(example)
    return speech

def predict(example):
    return tokenizer.batch_decode(torch.argmax(model(tokenizer(prepare_example(example), return_tensors="pt").input_values).logits, dim=-1))

@app.route('/')
def index():
    flash(" Welcome to Aymen & Mohammed's site")
    return render_template('index.html')

@app.route('/audio_to_text/')
def audio_to_text():
    flash(" Press Start to start recording audio and press Stop to end recording audio")
    return render_template('audio_to_text.html')


@app.route('/audio', methods=['POST'])
def audio():
    with open(audio_file, 'wb') as f:
        f.write(request.data)
        f.close()
        text = predict(audio_file)
        print(text)
        if(text):
            return_text = " Did you say : <br> " + text[0] + " <br> " 
        else:
            return_text = " Sorry!!!! Voice not Detected "

    return str(return_text)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    flash("Press Browse to select an audio file from your PC and press recognize to see the prediction")
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        if not os.path.exists('upload'):
            os.makedirs('upload')
        file.save(os.path.join('upload', filename))
        return redirect(url_for('prediction', filename=filename))
    return render_template('upload.html')


@app.route('/predicted/<filename>')
def prediction(filename):
    file_to_predict = os.path.join('upload', filename)
    text = predict(file_to_predict)
    return_text = text[0] + "."
    os.remove(file_to_predict)

    return render_template('upload.html', Prediction=return_text)

if __name__ == "__main__":
    app.run(debug=True)