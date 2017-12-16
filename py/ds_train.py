from keras.models import load_model
from scipy import io
import numpy as np
from bb_feature import get_iou, get_v

fpath = '../../MOT17/train/'
foldername = ('MOT17-02-FRCNN', 'MOT17-04-FRCNN', 'MOT17-05-FRCNN',
              'MOT17-09-FRCNN', 'MOT17-10-FRCNN', 'MOT17-11-FRCNN',
              'MOT17-13-FRCNN')

v_model = load_model('v_model.h5')
p_model = load_model('p_model.h5')

for folder in foldername:
    fname = '%s%s/gt/gt2.mat' % (fpath, folder)
    bb_all = io.loadmat(fname)['dets']
    id_num = bb_all[:, 1].max()
    for id_cnt in range(1, id_num + 1):
        bb_id = bb_all[bb_all[:, 1] == id_cnt, :]
        if bb_id.shape[0] == 0:
            continue
        max_frame = bb_id[:, 0].max()
        min_frame=bb_id[:, 0].min()
        bb_list = np.zeros((7, 4))
        v_list = np.zeros((6, 2), dtype='float32')
        p_list = np.zeros((6, 1), dtype='float32')
        for f_cnt in range(min_frame, max_frame + 1):
            n = f_cnt - min_frame
            if n < 7:
                bb_list[n] = bb_id[n, 2:6]
                if n > 0:
                    v_list[n - 1] = get_v(bb_list[n - 1], bb_list[n])
                    p_list[n - 1] = get_iou(bb_list[n - 1], bb_list[n])
            else:
                current_bb = bb_id[n, 2:6]
                current_v = get_v(bb_list[6], current_bb)
                current_p = get_iou(bb_list[6], current_bb)
                v_loss = v_model.evaluate(x=v_list, y=current_v, batch_size=1, verbose=0)
