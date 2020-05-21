
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import argparse

def process_image(image):
    """
    Image shape is (224,224,3)
    Parameters to pass:
    image [type - np.array] - image
    Returns:
    image_processed [type - np.array] - reshaped image (224,224,3)
    """

    image_tensor = tf.convert_to_tensor(image)
    image_processed = tf.image.resize(image_tensor, size = (224,224))
    image_processed = image_processed / 255.
    image_final_processed = image_processed.numpy()

    return image_final_processed

def predict(image_path, model, top_k = 5):
    """
    To predict from model
    Parameters:
    image_path [type - str]     - String
    model [type - Keras.model]  - Keras model
    top_k [type - int]          - number of n-highest values to return, default = 5
    Returns:
    classes [type - nd.array]   - top_k classes predicted, not zero-indexed!
    probs   [type - nd.array]   - top_k probabilities

    """

    #read image
    im = Image.open(image_path)
    image = np.asarray(im)

    #process image
    image_processed = process_image(image)
    image_processed = np.expand_dims(image_processed, axis = 0)

    #predict
    predictions = model.predict(image_processed)

    classes = np.argsort(-predictions[0])[:top_k]
    probs = np.asarray([predictions[0][classes]]).reshape(-1)
   
    #for index from 1
    classes = classes + 1

    return probs, classes