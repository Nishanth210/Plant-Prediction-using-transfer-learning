import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import tensorflow as tf
from tensorflow.keras.preprocessing import image

from flask import Flask, render_template, request
import numpy as np
from keras.preprocessing import image
import io
from PIL import Image
import tensorflow.keras.utils as image
from flask import Flask, render_template,request
import pickle
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

app = Flask(__name__)
from keras.models import load_model

model = load_model('my_model.h5')

# with open('trans1', 'rb') as file:
#     model = pickle.load(file)
# # Load the Keras model
# model = tf.keras.models.load_model('trans1')

from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename
 
app = Flask(__name__)
 
UPLOAD_FOLDER = 'static/uploads/'
 
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
     
 
@app.route('/')
def home():
    return render_template('PlantPredict.html')

@app.route('/about')
def about():
    return render_template('About.html')
 
@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        test_image =image.load_img(file_path,target_size = (64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = model.predict(test_image)
        
        res=result[0]
        val=max(res)
        ind = np.where(res == val)[0]
        class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Corn_(maize)___Common_rust_']  # Replace with your class names
        predicted_class_name = class_names[int(ind)]
        return render_template('PlantPredict.html', prediction_text='Disease for the given image is: {}'.format(predicted_class_name))
    
 
@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__=='__main__':
    app.run(debug=True)