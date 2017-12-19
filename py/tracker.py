import numpy as np

fpath = '../../MOT17/test/'
foldername = ('MOT17-01-FRCNN', 'MOT17-03-FRCNN', 'MOT17-06-FRCNN',
              'MOT17-07-FRCNN', 'MOT17-08-FRCNN', 'MOT17-12-FRCNN',
              'MOT17-14-FRCNN')
resolution = ((1920, 1080), (1920, 1080), (640, 480), (1920, 1080),
              (1920, 1080), (1920, 1080), (1920, 1080))
length = (450, 1500, 1194, 500, 625, 900, 750)

threshold_l = 0  # low detection threshold
threshold_s = 0  # score threshold

id_active, id_inactive = [], []

for folder, res, l in zip(foldername, resolution, length):
    fname_det = '%s%s/det/det.txt' % (fpath, folder)
    dets = np.loadtxt(fname_det, delimiter=',')
    dets = dets.astype('float32')
    for f_num in range(1, l + 1):
        dets_f = dets[dets[:, 0] == f_num, :]
        dets_f = dets_f[dets_f[:, -1] > threshold_l, :]
        if dets_f.shape[0] == 0:
            continue
        
        id_updated  = []
        for id_ in id_active:
            # matches the bb with the highest score
            bb = 0
            if ds_score >= threshold_s:
                id_ # append bb with this score

        # creates new ids
        id_new = [{'bb': np.array([det[2:6]]),
                   'v_list': np.zeros((6, 2), dtype='float32'),
                   'p_list': np.zeros((6, 1), dtype='float32'),
                   'max_score': det[-1],
                   'f_start': f_num} for det in dets_f]
