'''motion model
'''

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras import optimizers
from keras import regularizers
from scipy import io

LRATE, DECAY = 1e-3, 2e-5
OPTIMIZER, LOSS = 'adam', 'mean_squared_error'
DIMS, LENGTH, LSTM_UNITS = 2, 6, 20
EPOCHS, BATCH_SIZE, VERBOSE = 50, 64, 2
FPATH = '../../train/v/%s.mat'

x_train = io.loadmat(FPATH % 'v_x_train')['v_x_train']
x_test = io.loadmat(FPATH % 'v_x_test')['v_x_test']
y_train = io.loadmat(FPATH % 'v_y_train')['v_y_train']
y_test = io.loadmat(FPATH % 'v_y_test')['v_y_test']

model = Sequential()
model.add(LSTM(LSTM_UNITS, input_shape=(LENGTH, DIMS),
               kernel_regularizer=regularizers.l2(1e-3),
               recurrent_regularizer=regularizers.l2(1e-3), unroll=True))
model.add(Dense(DIMS))

m_adam = optimizers.Adam(lr=LRATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08,
                         decay=DECAY)
model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=['accuracy'])
model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
          verbose=VERBOSE, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)

model.save('m_model.h5')
