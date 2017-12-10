import keras
from keras.layers import Input, Dense
from keras.models import Model
from keras.applications.vgg16 import VGG16

target_a = Input(shape=(224, 224, 3))
target_b = Input(shape=(224, 224, 3))

shared_vgg16 = VGG16(include_top=True, weights='imagenet')
