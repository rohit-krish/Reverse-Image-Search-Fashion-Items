import pickle
from tensorflow import keras
import numpy as np
import cv2
from mylib.show import stackIt

feature_list = pickle.load(open('./embeddings.pkl', 'rb'))
filenames = pickle.load(open('./filenames.pkl', 'rb'))

image = keras.preprocessing.image
resnet50 = keras.applications.resnet50

model = resnet50.ResNet50(
    weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = keras.Sequential([
    model,
    keras.layers.GlobalAveragePooling2D()
])


def norm(pred):
    return np.sqrt(np.dot(pred, pred))


USE_KNN = True

img_path = './test/1553.jpg'
n_neighbors = 5

img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array, axis=0)
preprocessed_img = resnet50.preprocess_input(expanded_img_array)
result = model.predict(preprocessed_img).flatten()
normalized_result = result / norm(result)

if USE_KNN:
    from sklearn.neighbors import NearestNeighbors
    neibhbors = NearestNeighbors(
        n_neighbors=n_neighbors, algorithm='brute', metric='euclidean'
    )  # cosine
    neibhbors.fit(feature_list)
    distances, indices = neibhbors.kneighbors([normalized_result])

    images = []

    for i in indices[0]:
        images.append(cv2.imread(filenames[i]))

    images = np.array(images)[None, ...]

else:
    from annoy import AnnoyIndex
    annoy_index = AnnoyIndex(2048, 'angular')
    annoy_index.load('./embeddings.ann')
    nns = annoy_index.get_nns_by_vector(normalized_result, n_neighbors)

    images = []
    for i in nns:
        images.append(cv2.imread(filenames[i]))
    
    images = np.array(images)[None, ...]


cv2.imshow('results', stackIt(images))
cv2.imshow('original', cv2.imread(img_path))
cv2.waitKey(0)
