import pickle
from annoy import AnnoyIndex

features = pickle.load(open('../embeddings.pkl', 'rb'))

annoy_index = AnnoyIndex(2048, 'angular')

for i, feat in enumerate(features):
    annoy_index.add_item(i, feat)

annoy_index.build(15)
annoy_index.save('../embeddings.ann')
