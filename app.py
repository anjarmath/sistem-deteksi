import os
from flask import Flask, render_template, request, jsonify, url_for, send_file, send_from_directory
from keras.utils import load_img
from keras.utils import img_to_array
import tensorflow as tf
import numpy as np
from PIL import Image
import shutil

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = './img/'



target_names_c19 = ['COVID','Normal','Viral Pneumonia']
target_names_pneumonia = ['Normal','Viral Pneumonia']

def load_model(service, modelname):
    model = tf.keras.models.load_model('./model/{}/model_{}.h5'.format(service, modelname))
    if (service == 'c19'):
        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer='SGD',
            metrics=['accuracy']
        )
    else:
        model.compile(
            loss='binary_crossentropy',
            optimizer='Adam',
            metrics=['accuracy']
        )
    return model

def is_grey_scale(img_path):
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    for i in range(w):
        for j in range(h):
            r, g, b = img.getpixel((i,j))
            if r != g != b: 
                return False
    return True

@app.route('/', methods = ['GET'])
def detection_system():
    shutil.rmtree('./img/')
    os.mkdir('./img/')
    return render_template("index.html")

@app.route('/c19', methods = ['GET'])
def detection_c19():
    return render_template("c19/index.html")

@app.route('/pneumonia', methods = ['GET'])
def detection_pneumonia():
    return render_template("pneumonia/index.html")

@app.route('/<service>/predict', methods = ['POST'])
def predict(service):
    imagefile = request.files['imagefile']
    modelname = request.values['architecture']
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], imagefile.filename)
    imagefile.save(image_path)

    image = load_img(image_path, target_size=(224,224))
    if (is_grey_scale(image_path) == True):
        image1 = img_to_array(image)
        image1 = np.expand_dims(image1, axis=0)
        image1 = np.vstack([image1])
        model = load_model(service, modelname)
        prediksi = model.predict(image1)
        skor = np.max(prediksi)
        classes = np.argmax(prediksi)
        if skor > 0.9:
            if (service == 'c19'):
                hasil = target_names_c19[classes]
            else:
                hasil = target_names_pneumonia[classes]
        else:
            hasil='Tidak terdeteksi apapun, periksa gambar Anda'
    else:
        hasil = 'Gambar tidak terdeteksi sebagai citra x-ray'

    return render_template(service+"/hasil.html", result=hasil, img=imagefile.filename)

@app.route('/img/<fileimg>')
def send_uploaded_image(fileimg=''):
    return send_from_directory( app.config['UPLOAD_FOLDER'], fileimg)

if __name__ == '__main__':
    app.run(port=8000, debug=True)
