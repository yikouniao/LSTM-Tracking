from keras.models import load_model
from scipy import io
from scipy import misc
import numpy as np
from skimage.transform import resize
from num2fname import num2fname

in_fpath = '../../MOT17/train/'
out_fpath = '/media/lym/Elements/MOT17/train/'
foldername = ('MOT17-02-FRCNN', 'MOT17-04-FRCNN', 'MOT17-05-FRCNN',
              'MOT17-09-FRCNN', 'MOT17-10-FRCNN', 'MOT17-11-FRCNN',
              'MOT17-13-FRCNN')
filename = ('x1_bb_splice', 'x2_bb_splice')

width, height, channel = 52, 170, 3

bottom_vgg = load_model('bottom_vgg.h5')

for folder in foldername: 
    for file in filename:
        for i in range(1000):
            bb_fname = '%s%s/a/%s_%s.mat' % (in_fpath, folder, file,
                                             str(i + 1))
            bb = io.loadmat(bb_fname)[file]
            n = bb.shape[0]
            x_a = np.zeros((n, 42, 13, 128), dtype='float32')
            for j in range(n // 32):
                img_batch = np.zeros((32, height, width, channel),
                                     dtype='float32')
                for k in range(32):
                    row_num = j * 32 + k
                    img_fname = '%s%s/img1/%s' % (in_fpath, folder,
                                                  num2fname(bb[row_num][0]))
                    img = misc.imread(img_fname)
                    img = img[bb[row_num][3]:bb[row_num][3] + bb[row_num][5],
                              bb[row_num][2]:bb[row_num][2] + bb[row_num][4]]
                    img = resize(img, (height, width, channel), mode='reflect')
                    #misc.imsave('temp.jpg', img)
                    img = img.astype('float32')
                    img_batch[k] = img

                np.save('img_batch.npy', img_batch)
                x_a[j * 32:(j + 1) * 32] = bottom_vgg.predict(
                    x=img_batch, batch_size=32, verbose=0)
            
            out_fname = '%s%s/a/%s_%s.npy' % (out_fpath, folder, file,
                                              str(i + 1))
            np.save(out_fname, x_a)
