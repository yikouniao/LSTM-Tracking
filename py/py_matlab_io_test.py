# def mysum(x):
#     """my sum
#     for io test between python and matlab
#     """
    
#     print(x[0])
#     return x[1]



# a=[1,2,3]
# aa=mysum(a)
# print(aa)

# import scipy
from scipy import io
# x = io.loadmat('../x.mat')['x']
# print(x.shape)
# print(x[0])
# print(x[0][0])
# print(x[0][0][0])

foldernames = ['MOT17-02-FRCNN', 'MOT17-04-FRCNN', 'MOT17-05-FRCNN',
              'MOT17-09-FRCNN', 'MOT17-10-FRCNN', 'MOT17-11-FRCNN',
              'MOT17-13-FRCNN']

for fname in foldernames:
    fpath = '../../train/%s/v/v_x_train.mat' % fname
    x = io.loadmat(fpath)['idv_x_train']
