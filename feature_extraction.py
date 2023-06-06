import tensorflow as tf
from tensorflow import keras
from keras.layers import GlobalAveragePooling2D
from keras.applications.resnet import ResNet50, preprocess_input

import numpy as np
import os
import pickle

model = ResNet50(
    include_top=False, weights='imagenet', input_shape=(224, 224, 3)
)
model.trainable = False

model = tf.keras.Sequential([
    model,
    GlobalAveragePooling2D()
])

def norm(pred):
    '''
    l2 norm
    np.linalg.norm
    '''
    return np.sqrt(np.dot(pred, pred))

def extract_featres(img_path, model):
    img = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = keras.preprocessing.image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    pred = model.predict(preprocessed_img).flatten()

    return pred / norm(pred)  # normalized


file_names = np.array(['./images/'+fn for fn in os.listdir('./images')])

feature_list = np.array([extract_featres(fn, model) for fn in file_names])

pickle.dump(feature_list, open('./embeddings.pkl', 'wb'))
pickle.dump(file_names, open('filenames.pkl', 'wb'))
