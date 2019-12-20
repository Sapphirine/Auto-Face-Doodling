from flask import Flask, render_template, request
from scipy.misc import imread, imresize
import numpy as np
import re
import sys
import os

sys.path.append(os.path.abspath("./model"))
sys.path.append(os.path.abspath("./country_model"))

from load import *

app = Flask(__name__)
global model, graph
model, graph = init()

model_face = load_face()
model_eye = load_eye()
model_mouth = load_mouth()
model_nose = load_nose()
model_eye = load_eye()


import base64

classes = ['ear','eye','face','mouth','nose']
countries = ['AU','BR','KR','SA']
# decoding an image from base64 into raw representation
def convertImage(imgData1):
    imgstr = re.search(r'base64,(.*)', str(imgData1)).group(1)
    with open('output.png', 'wb') as output:
        output.write(base64.b64decode(imgstr))


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    imgData = request.get_data()
    # encode it into a suitable format
    convertImage(imgData)
    # read the image into memory
    x = imread('output.png', mode='L')
    # make it the right size
    x1 = imresize(x, (28, 28))
    x1 = x1.reshape(1, 28, 28, 1)
    x2 = imresize(x, (125, 125))
    x2 = x2.reshape(1, 125, 125, 1)
    # convert to a 4D tensor to feed into our model
    
    # in our computation graph
    with graph.as_default():
        # perform the prediction
        out = model.predict(x1)
        #print(out)
        facial = np.argmax(out[0])
        #print(np.argmax(out, axis=1))
        
        #print(classes[facial])
        if facial == 0: #face
            out2 = model_face.predict(x2)
        if facial == 1: #eye
            out2 = model_eye.predict(x2)
        if facial == 2: #mouth
            out2 = model_mouth.predict(x2)
        if facial == 3: #nose
            out2 = model_nose.predict(x2)
        if facial == 4: #ear
            out2 = model_ear.predict(x2)
        # convert the response to a string
        
        country = np.argmax(out2)
        result = "It's " + "'" + classes[facial] + "'"+ " from " + "'" + countries[country] + "'"
        print(result)
        return result


if __name__ == "__main__":
    # run the app locally on the given port
    app.run(host='0.0.0.0', port=9091)
