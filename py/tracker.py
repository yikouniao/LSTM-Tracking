import numpy as np
from operator import itemgetter
from time import time
from bb_proc import get_iou, bb_update_vp2, ds_score, bb_update_vp, bb_pred

# FRCNN test
fpath = '../../MOT17/test/'
foldername = ('MOT17-01-FRCNN', 'MOT17-03-FRCNN', 'MOT17-06-FRCNN',
              'MOT17-07-FRCNN', 'MOT17-08-FRCNN', 'MOT17-12-FRCNN',
              'MOT17-14-FRCNN')
resolution = ((1920, 1080), (1920, 1080), (640, 480), (1920, 1080),
              (1920, 1080), (1920, 1080), (1920, 1080))
length = (450, 1500, 1194, 500, 625, 900, 750)

# FRCNN train
# fpath = '../../MOT17/train/'
# foldername = ('MOT17-02-FRCNN', 'MOT17-04-FRCNN', 'MOT17-05-FRCNN',
#               'MOT17-09-FRCNN', 'MOT17-10-FRCNN', 'MOT17-11-FRCNN',
#               'MOT17-13-FRCNN')
# resolution = ((1920, 1080), (1920, 1080), (640, 480), (1920, 1080),
#               (1920, 1080), (1920, 1080), (1920, 1080))
# length = (600, 1050, 837, 525, 654, 900, 750)

threshold_l = 0  # low detection threshold
threshold_h = 0.9  # high detection threshold
threshold_s = 0.0377  # score threshold
threshold_s2 = 0.4  # score threshold for id shorter than 7 frames
t_min = 4  # time threshold


# SDP test
# fpath = '../../MOT17/test/'
# foldername = ('MOT17-01-SDP', 'MOT17-03-SDP', 'MOT17-06-SDP',
#               'MOT17-07-SDP', 'MOT17-08-SDP', 'MOT17-12-SDP',
#               'MOT17-14-SDP')
# resolution = ((1920, 1080), (1920, 1080), (640, 480), (1920, 1080),
#               (1920, 1080), (1920, 1080), (1920, 1080))
# length = (450, 1500, 1194, 500, 625, 900, 750)

# SDP train
# fpath = '../../MOT17/train/'
# foldername = ('MOT17-02-SDP', 'MOT17-04-SDP', 'MOT17-05-SDP',
#               'MOT17-09-SDP', 'MOT17-10-SDP', 'MOT17-11-SDP',
#               'MOT17-13-SDP')
# resolution = ((1920, 1080), (1920, 1080), (640, 480), (1920, 1080),
#               (1920, 1080), (1920, 1080), (1920, 1080))
# length = (600, 1050, 837, 525, 654, 900, 750)

# threshold_l = 0.3  # low detection threshold
# threshold_h = 0.5  # high detection threshold
# threshold_s = 0.0359  # score threshold
# threshold_s2 = 0.3 # score threshold for id shorter than 7 frames
# t_min = 5  # time threshold


# DPM test
# fpath = '../../MOT17/test/'
# foldername = ('MOT17-01-DPM', 'MOT17-03-DPM', 'MOT17-06-DPM',
#               'MOT17-07-DPM', 'MOT17-08-DPM', 'MOT17-12-DPM',
#               'MOT17-14-DPM')
# resolution = ((1920, 1080), (1920, 1080), (640, 480), (1920, 1080),
#               (1920, 1080), (1920, 1080), (1920, 1080))
# length = (450, 1500, 1194, 500, 625, 900, 750)

# DPM train
# fpath = '../../MOT17/train/'
# foldername = ('MOT17-02-DPM', 'MOT17-04-DPM', 'MOT17-05-DPM',
#               'MOT17-09-DPM', 'MOT17-10-DPM', 'MOT17-11-DPM',
#               'MOT17-13-DPM')
# resolution = ((1920, 1080), (1920, 1080), (640, 480), (1920, 1080),
#               (1920, 1080), (1920, 1080), (1920, 1080))
# length = (600, 1050, 837, 525, 654, 900, 750)

# threshold_l = -10  # low detection threshold
# threshold_h = -9  # high detection threshold
# threshold_s = 0.0155  # score threshold
# threshold_s2 = 0.36  # score threshold for id shorter than 7 frames
# t_min = 7  # time threshold


time_cnt = 0

for folder, res, l in zip(foldername, resolution, length):
    print('Processing %s...' % folder)
    fname_det = '%s%s/det/det.txt' % (fpath, folder)
    dets = np.loadtxt(fname_det, delimiter=',')
    dets = dets.astype('float32')
    start = time()

    id_active, id_inactive = [], []

    for f_num in range(1, l + 1):
        dets_f = dets[dets[:, 0] == f_num, :]
        dets_f = dets_f[dets_f[:, 6] > threshold_l, :]
        if dets_f.shape[0] == 0:
            continue
        
        match_scores = np.zeros(dets_f.shape[0], dtype='float32')
        matched_flag = np.zeros(dets_f.shape[0], dtype=bool)
        id_updated  = []
        for id_ in id_active:
            # if this id is too short to use lstm
            if len(id_['bb']) < 7:
                # calculates the bb matching score
                for det_num, det in enumerate(dets_f):
                    if matched_flag[det_num] == True:
                        match_scores[det_num] = 0
                    else:
                        match_scores[det_num] = get_iou(
                            id_['bb'][-1], det[2:6])[0][0]
                best_match = dets_f[match_scores.argmax()]
                best_match_score = match_scores.max()

                # matches the bb with highest score
                if best_match_score >= threshold_s2:
                    bb_update_vp2(id_, best_match[2:6], res)
                    id_['bb'].append(best_match[2:6])
                    id_['max_score'] = max(id_['max_score'], best_match[-1])
                    matched_flag[match_scores.argmax()] = True
                    id_updated.append(id_)
                
                # finishes this id
                else:
                    # if it's a valid id
                    if (id_['max_score'] >= threshold_h and
                        len(id_['bb']) >= t_min):
                        id_inactive.append(id_)

            else:
                # calculates the bb matching score
                for det_num, det in enumerate(dets_f):
                    if matched_flag[det_num] == True:
                        match_scores[det_num] = 0
                    else:
                        match_scores[det_num] = ds_score(
                            id_, det[2:6], res)[0][0]
                best_match = dets_f[match_scores.argmax()]
                best_match_score = match_scores.max()

                # matches the bb with highest score
                if best_match_score >= threshold_s:
                    bb_update_vp(id_, best_match[2:6], res)
                    id_['bb'].append(best_match[2:6])
                    id_['max_score'] = max(id_['max_score'], best_match[-1])
                    matched_flag[match_scores.argmax()] = True
                    id_['pred'] = 0
                    id_updated.append(id_)

                # the id was not updated, predict the next bb
                elif id_['pred'] < 6:
                    # not updating max_score here
                    bb_pred(id_, res)
                    id_updated.append(id_)

                # finishes this id
                else:
                    # if it has pred, clear all pred. not clearing v, p list
                    for i in range(id_['pred']):
                        id_['bb'].pop()
                    id_['pred'] = 0

                    # if it's a valid id
                    if (id_['max_score'] >= threshold_h and
                        len(id_['bb']) >= t_min):
                        id_inactive.append(id_)

        # creates new ids
        id_new = [{'bb': [det[2:6]],
                   'v_list': np.zeros((6, 2), dtype='float32'),
                   'p_list': np.zeros((6, 1), dtype='float32'),
                   'max_score': det[6],
                   'f_start': f_num,
                   'pred': 0}
                   for det_num, det in enumerate(dets_f)
                   if matched_flag[det_num] == False]
        id_active = id_updated + id_new

    # finishes the remained ids
    for id_ in id_active:
        # if it has pred, clear all pred. not clearing v, p list
        for i in range(id_['pred']):
            id_['bb'].pop()
        id_['pred'] = 0

        # if it's a valid id
        if id_['max_score'] >= threshold_h and len(id_['bb']) >= t_min:
            id_inactive.append(id_)
    
    end = time()
    time_cnt += end - start
    # now id_inactive is the final result
    result_bb = []
    for id_num, id_ in enumerate(id_inactive):
        for bb_num, bb in enumerate(id_['bb']):
            result_bb += [[id_['f_start'] + bb_num, id_num + 1, bb[0], bb[1],
                           bb[2], bb[3], -1, -1, -1, -1]]
    result_bb.sort(key=itemgetter(1, 0))
    with open('./results/%s.txt' % folder, 'w') as rst_f:
        for bb in result_bb:
            rst_f.write(','.join([str(value) for value in bb]) + '\n')

print('Total tracking time consumption:', time_cnt, 's.')
