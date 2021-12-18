import json
import numpy as np
from annoy import AnnoyIndex
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

TEST_IMAGE = 'test_img.jpg'

NUM_OF_MATCHES = 10


def index_features(features, n_trees=1000, dims=4096, is_dict=False):
    """
    Use Annoy to index our features to be able to query them rapidly
    :param features: array of item features
    :param n_trees: number of trees to use for Annoy. Higher is more precise but slower.
    :param dims: dimension of our features
    :return: an Annoy tree of indexed features
    """
    feature_index = AnnoyIndex(dims, metric='angular')
    for i, row in enumerate(features):
        vec = row
        if is_dict:
            vec = features[row]
        feature_index.add_item(i, vec)
    feature_index.build(n_trees)
    return feature_index

images_features = np.load('image_features.npy')

image_index = index_features(images_features)


pretrained_vgg16 = VGG16(weights='imagenet', include_top=True)
model = Model(inputs=pretrained_vgg16.input,
                  outputs=pretrained_vgg16.get_layer('fc2').output)

def predict_similar_images(img_path, model, image_index, num_matches = 10):  
    img = image.load_img(img_path, target_size=(224, 224))
    x_raw = image.img_to_array(img)
    x_expand = np.expand_dims(x_raw, axis=0)
    x = preprocess_input(x_expand)
    featured = model.predict(x)
    index_of_file = image_index.get_nns_by_vector(featured.reshape(-1), num_matches )
    
    return index_of_file

out_come = predict_similar_images(TEST_IMAGE, model, image_index, num_matches = NUM_OF_MATCHES)

with open('file_index.json') as f_in:
    file_index = json.load(f_in)
    
    with open('name_to_url.json') as f_in:
        name_to_url = json.load(f_in)
        for outt in out_come:
            print(name_to_url[file_index[str(outt)]])
