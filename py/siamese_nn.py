
# import keras
from keras.layers import Input, Flatten, concatenate, Dense
from keras.models import Model
from keras import optimizers

input_shape = (42, 13, 128)
learning_rate, learning_rate_decay = 1e-3, 5e-6
dropout_rate = 0.4

target_a = Input(shape=input_shape)
target_b = Input(shape=input_shape)

flatten_a = Flatten()(target_a)
flatten_b = Flatten()(target_b)

merged_vector = concatenate([flatten_a, flatten_b], axis=-1)
#dropouted = Dropout(rate=dropout_rate)(merged_vector)
predictions = Dense(1, activation='sigmoid')(merged_vector)

model = Model(inputs=[target_a, target_b], outputs=predictions)

my_adam = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999,
                          epsilon=1e-08, decay=learning_rate_decay)
model.compile(optimizer=my_adam, loss='binary_crossentropy',
              metrics=['accuracy'])
# history = model.fit(x_train, y_train, epochs=fit_epochs,
#                     batch_size=fit_batch_size, verbose=fit_verbose,
#                     validation_data=(x_test, y_test))
# training_plot(history)
model.summary()
model.save('siamese.h5')