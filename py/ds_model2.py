import numpy as np
from sklearn.svm import SVC
import pickle

fpath = '../../MOT17/train/'

ds_x = np.load('%sds_x.npy' % fpath)
ds_y = np.load('%sds_y.npy' % fpath)
length = 102400
ds_x_train, ds_x_test = ds_x[:length], ds_x[length:length * 2]
ds_y_train, ds_y_test = ds_y[:length], ds_y[length:length * 2]

ds_model = SVC(C=1.0, kernel='rbf', gamma='auto', coef0=0.0, probability=False,
    shrinking=True, tol=1e-3, cache_size=1024, class_weight=None, verbose=False,
    max_iter=-1, decision_function_shape='ovr', random_state=None)
ds_model.fit(ds_x_train, ds_y_train)
print(ds_model.score(ds_x_test, ds_y_test))
print(ds_model.predict([[0.9, 0.005], [0.05, 0.005]]))

with open('ds_model.pickle', 'wb') as fw:
    pickle.dump(ds_model, fw)

with open('ds_model.pickle', 'rb') as fr:
    new_svm = pickle.load(fr)
    print(new_svm.predict([[0.5, 0.005], [0.2, 0.005]]))
