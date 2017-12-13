from keras.applications.vgg16 import VGG16
from keras.models import Model

width, height, channel = 52, 170, 3

bottom_vgg = VGG16(include_top=False, weights='imagenet',
                   input_shape=(height, width, channel))

for i in range(12):
    bottom_vgg.layers.pop()

# output vector size: (None, 42, 13, 128)
feature = Model(inputs=bottom_vgg.input, outputs=bottom_vgg.layers[-1].output)

for layer in feature.layers:
    layer.trainable = False

feature.compile(optimizer='Adam', loss='mean_squared_error',
                metrics=['accuracy'])
feature.summary()
feature.save('bottom_vgg.h5')