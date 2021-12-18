
import numpy as np
import os
import time
import json

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image



def load_image_data(folder, file_index = {} ):
    
    if not file_index:
        names = [fold for fold in os.listdir(folder) if ".DS" not in fold]
        
    else:
        names = [fold for fold in os.listdir(folder) if ".DS" not in fold]
        names = list(set(names) - set(file_index.values()))
    
    image_list = []
    images_names = []
    
    


    for name in names:
        full_path = os.path.join(folder, name)
        try:
            img = image.load_img(full_path, target_size=(224, 224))
            images_names.append(name)
        except:
            
            continue
            
        x_raw = image.img_to_array(img)
        x_expand = np.expand_dims(x_raw, axis=0)
        x = preprocess_input(x_expand)
        image_list.append(x)
        
        
    img_data = np.array(image_list)
    img_data = np.rollaxis(img_data, 1, 0)
    img_data = img_data[0]

    return img_data, images_names


try:    
    with open('file_index.json') as f_in:
        file_index = json.load(f_in)
        
except:
    file_index = {}
    

new_images,  new_images_names = load_image_data('images_base', file_index )

pretrained_vgg16 = VGG16(weights='imagenet', include_top=True)
model = Model(inputs=pretrained_vgg16.input,
                  outputs=pretrained_vgg16.get_layer('fc2').output)



def generate_features(images_names, model, file_index, previous_feature_array):
    """
    Takes in an array of image paths, and a trained model.
    Returns the activations of the last layer for each image
    :param images_names: array of image names
    :param model: pre-trained model
    :return: array of last-layer activations, and mapping from array_index to file_path
    """
    start = time.time()
    
    
    images = np.zeros(shape=(len(images_names), 224, 224, 3))
    
    file_index_length = len(file_index)
    
    folder = 'images_base'
    
    for i, f in enumerate(images_names):
        
        file_index[i + file_index_length] = f
            
        full_path = os.path.join(folder, f)
        img = image.load_img(full_path, target_size=(224, 224))
        x_raw = image.img_to_array(img)
        x_expand = np.expand_dims(x_raw, axis=0)
        images[i, :, :, :] = x_expand

    
    inputs = preprocess_input(images)
    
    if previous_feature_array.size > 0:
        new_images_features = model.predict(inputs)
        images_features = np.vstack((previous_feature_array, new_images_features))
        
    else:
        images_features = model.predict(inputs)
    
    end = time.time()
    print("Inference done, %s Generation time" % (end - start))
    return images_features, file_index




try:
    previous_images_features = np.load('image_features.npy')
except:
    previous_images_features = np.array([])

all_images_features, latest_file_index = generate_features(new_images_names, model, file_index, previous_images_features)

np.save('image_features', all_images_features)

with open('file_index.json', 'w') as fp:
    json.dump(latest_file_index, fp)