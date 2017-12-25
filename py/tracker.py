import numpy as np
from bb_proc import ds_score

fpath = '../../MOT17/test/'
foldername = ('MOT17-01-FRCNN', 'MOT17-03-FRCNN', 'MOT17-06-FRCNN',
              'MOT17-07-FRCNN', 'MOT17-08-FRCNN', 'MOT17-12-FRCNN',
              'MOT17-14-FRCNN')
resolution = ((1920, 1080), (1920, 1080), (640, 480), (1920, 1080),
              (1920, 1080), (1920, 1080), (1920, 1080))
length = (450, 1500, 1194, 500, 625, 900, 750)

threshold_l = 0  # low detection threshold
threshold_h = 0  # high detection threshold
threshold_s = 0  # score threshold
t_min = 0 # time threshold

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
        
        match_scores = np.zeros(dets_f.shape[0], dtype='float32')
        matched_flag = np.zeros(dets_f.shape[0], dtype=bool)
        id_updated  = []
        for id_ in id_active:
            # calculates the bb matching score
            for det, m_scores, flag in zip(dets_f, match_scores, matched_flag):
                if flag == True:
                    m_scores = 0
                else:
                    m_scores = ds_score(id_, det, res)
            best_match = dets_f[match_scores.argmax()]
            best_match_score = match_scores.max()

            # matches the bb with highest score
            if best_match_score >= threshold_s:
                id_['bb'].append(best_match[2:6])
                id_['max_score'] = max(id_['max_score'], best_match[-1])
                id_updated.append(id_)
                matched_flag[match_scores.argmax()] = True
                id_['pred'] = 0

            # the id was not updated, predict the next bb
            elif id_['pred'] < 6:
                # not updating max_score here

                id_['pred'] += 1

            # finishes this id
            else:
                # if it has pred, clear all pred. not clearing v, p list
                for i in range(id_['pred']):
                    id_['bb'].pop()
                id_['pred'] = 0

                # if it's a valid id
                if id_['max_score'] >= threshold_h and len(id_['bb']) >= t_min:
                    id_inactive.append(id_)

        # creates new ids
        id_new = [{'bb': [det[2:6]],
                   'v_list': np.zeros((6, 2), dtype='float32'),
                   'p_list': np.zeros((6, 1), dtype='float32'),
                   'max_score': det[-1],
                   'f_start': f_num,
                   'pred': 0} for det in dets_f]
