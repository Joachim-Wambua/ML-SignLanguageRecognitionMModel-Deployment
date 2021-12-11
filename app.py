from flask import Flask, render_template, request
from keras.models import load_model 
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
import cv2

# try:
#     import shutil
#     shutil.rmtree('uploaded')
#     '% cd uploaded % mkdir image % cd ..'
#     print()
# except:
#     pass

app = Flask(__name__)


optimised_model=tf.keras.models.load_model('ASL_Optimized.model')

alphabets_asl = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 
                 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 
                 'W', 'X', 'Y', 'Z', 'del', 'space', 'nothing']


def prep_image(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (64, 64))
    img=img.reshape(-1, 64, 64, 1)
    img=img.astype('float32')/255.0
    return  img

# Function to perform predictions
def predict_sign_lang(path):
    prediction = optimised_model.predict([prep_image(path)])
    label = np.argmax(prediction[0])
    return alphabets_asl[label]


@app.route('/')
# Render the home page
def home():
    return render_template('index.html')


@app.route('/submit', methods=['GET', 'POST'])
def predict():
    result = ''
    image_path = ''
    if request.method == 'POST':
        img = request.files['file']

        image_path = 'static/uploaded/' + img.filename
        img.save(image_path)

        result = predict_sign_lang(image_path)

    return render_template("index.html", prediction ='ASL Sign represents Letter: {}'.format(str(result.capitalize())), img_path = image_path)


if __name__ == '__main__':
    app.run(debug=True)
