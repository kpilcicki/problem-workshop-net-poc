# import the necessary packages
# from keras.applications import ResNet50
# from keras.preprocessing.image import img_to_array
# from keras.applications import imagenet_utils
import tensorflow as tf
# from PIL import Image
import numpy as np
import flask
import io
import os

import tf_utils as tfu
from predict import predict_with

FULL_MODEL_DIR = 'adagrad-dnn-40'
ADAM_MODEL_DIR = 'adam-dnn-40'

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
# model = None
predictor = None
adam_predictor = None

def load_model():
    global predictor
    with tf.Session() as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], FULL_MODEL_DIR)
        predictor = tf.contrib.predictor.from_saved_model(FULL_MODEL_DIR)

def load_adam_model():
    global adam_predictor
    with tf.Session() as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], ADAM_MODEL_DIR)
        adam_predictor = tf.contrib.predictor.from_saved_model(ADAM_MODEL_DIR)

@app.route("/predict", methods=["POST"])
def predict():
    return predict_with(predictor)

@app.route("/predict/adam", methods=["POST"])
def predict_adam():
    return predict_with(adam_predictor)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    load_model()
    load_adam_model()
    port = int(os.getenv('PORT', 5000))
    app.run(port=port, host='0.0.0.0')