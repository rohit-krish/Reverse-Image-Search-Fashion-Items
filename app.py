import streamlit as st
from PIL import Image
from tensorflow import keras
import pickle
import numpy as np
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors


def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = resnet50.preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result


image = keras.preprocessing.image
resnet50 = keras.applications.resnet50

model = resnet50.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = keras.Sequential([
    model,
    keras.layers.GlobalAveragePooling2D()
])

feature_list = pickle.load(open('./embeddings.pkl', 'rb'))
filenames = pickle.load(open('./filenames.pkl', 'rb'))

LIMIT = 5

neibhbors = NearestNeighbors(n_neighbors=LIMIT, algorithm='brute', metric='euclidean')
neibhbors.fit(feature_list)

st.title('Fashion Recommender System')

uploaded_file = st.file_uploader('Choose an image')

if uploaded_file is not None:
    with open('./uploads/'+uploaded_file.name, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    st.image(Image.open(uploaded_file))
    features = feature_extraction('./uploads/'+uploaded_file.name, model)
    distances, indices = neibhbors.kneighbors([features])

    cols = st.columns(LIMIT)
    for idx, col in enumerate(cols):
        with col:
            st.image(filenames[indices[0][idx]])
