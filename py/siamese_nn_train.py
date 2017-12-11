import numpy as np

fpath = '../../MOT17/train/'
foldername = ('MOT17-02-FRCNN', 'MOT17-04-FRCNN', 'MOT17-05-FRCNN',
              'MOT17-09-FRCNN', 'MOT17-10-FRCNN', 'MOT17-11-FRCNN',
              'MOT17-13-FRCNN')
split_factor = 0.7

for folder in foldername:
    gt_fname = '%s%s/gt/gt.txt' % (fpath, folder)
    dets = np.loadtxt(gt_fname, delimiter=',')
    dets = dets[dets[:, 6] == 1 and dets[:, 7] == 1 and dets[:, 8] > 0.8]
    print(dets.shape())