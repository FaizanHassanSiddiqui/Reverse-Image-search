
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
from tensorflow.keras import models

from tensorflow.keras.preprocessing import image

from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import math
import numpy as np



INPUT_IMAGE = 'skirt2.jpg'


from tensorflow.keras.applications import VGG16


model = VGG16(weights='imagenet',
              include_top= True,
              
              )


img = image.load_img(INPUT_IMAGE, target_size=(224,224), )

x_raw = image.img_to_array(img)
x_expand = np.expand_dims(x_raw, axis=0)
x = preprocess_input(x_expand)
output = model.predict(x)

label = decode_predictions(output)

print("The name of the image is: ", label[0][0][1])