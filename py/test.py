# from keras.models import Sequential
# from keras.utils import plot_model
# from numpy import array

# from keras.models import load_model

# model = load_model('v_model.h5')
# plot_model(model, to_file='v_model.png', show_shapes=True,
#            show_layer_names=False, rankdir='LR')
# y = model.predict(x=[[[1],[1],[1],[1],[1],[1]],[[0],[0],[0],[0],[0],[0]]],
#                   batch_size=1,verbose=1)
# print(y)

# from keras.models import load_model
# import numpy as np

# bottom_vgg = load_model('bottom_vgg.h5')
# for i in range(10):
#     img_batch=np.load('img_batch1.npy')
#     print(img_batch.shape[0])
#     # img_batch = img_batch[0:3]
#     # np.save('img_batch1.npy', img_batch)
#     y = bottom_vgg.predict(
#         x=img_batch, batch_size=3, verbose=0)
#     print(img_batch[1][1][1][0])
#     print(y[1][1][1][0])

# from keras.layers import Input, Dense, Flatten
# from keras.models import Model
# import numpy as np

# x_test = np.array((((2, 2), (3, 3)), ((4, 5), (6, 7))))

# # the nn model
# # not channels-first
# x = Input(shape=(2, 2))
# inner = Dense(units=2, activation='sigmoid')(x)
# inner = Flatten()(inner)
# y = Dense(units=1, activation='sigmoid')(inner)

# model = Model(inputs=x, outputs=y)
# model.compile(optimizer='Adam', loss='mean_squared_error',
#               metrics=['accuracy'])
# model.summary()
# y = model.predict(x=x_test, batch_size=1, verbose=0)
# print(y)

# y_test = np.array(((10), (10)))
# score = model.evaluate(x_test, y_test, verbose=0)
# print(score)

from bb_feature import get_v
bb1 = [0, 0, 100, 100]
bb2 = [50, 20, 100, 100]
print(get_v(bb1, bb2))
