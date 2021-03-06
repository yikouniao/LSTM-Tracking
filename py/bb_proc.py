from keras.models import load_model
import numpy as np

scale_factor = 180

v_model = load_model('v_model.h5')
p_model = load_model('p_model.h5')
ds_model = load_model('ds_model.h5')

def get_iou(bb_a, bb_b):
    """
    calculates the intersection over union of two bounding boxes
    format of bb: array [x_left, y_top, width, height]
    returns the iou score: array [[iou]] with type float32
    """

    # turns the bb format into [x_left, y_top, x_right, y_bottom]
    bb1 = np.array([bb_a[0], bb_a[1], bb_a[0] + bb_a[2], bb_a[1] + bb_a[3]],
                   dtype='float32')
    bb2 = np.array([bb_b[0], bb_b[1], bb_b[0] + bb_b[2], bb_b[1] + bb_b[3]],
                   dtype='float32')

    # calculates the intersection
    inter_x_left = max([bb1[0], bb2[0]])
    inter_y_top = max([bb1[1], bb2[1]])
    inter_x_right = min([bb1[2], bb2[2]])
    inter_y_bottom = min([bb1[3], bb2[3]])

    # checks if the iou is 0
    if inter_x_left >= inter_x_right or inter_y_top >= inter_y_bottom:
        return np.array([[0]], dtype='float32')

    # calculates the iou score
    inter_area = ((inter_x_right - inter_x_left) *
                  (inter_y_bottom - inter_y_top))
    area_1 = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    area_2 = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])
    union_area = area_1 + area_2 - inter_area
    return np.array([[inter_area / union_area]], dtype='float32')


def get_v(bb1, bb2, resolution):
    """
    calculates the velocity from bb1 to bb2
    format of bb: array [x_left, y_top, width, height]
    returns velocity: array [[x_v, y_v]] with type float32
    """

    # gets the center position
    c1 = (bb1[0] + bb1[2] / 2, bb1[1] + bb1[3] / 2)
    c2 = (bb2[0] + bb2[2] / 2, bb2[1] + bb2[3] / 2)
    return np.array([[(c2[0] - c1[0]) * scale_factor / resolution[0],
                      (c2[1] - c1[1]) * scale_factor / resolution[1]]],
                    dtype='float32')


def ds_score(id_, bb, resolution):
    """
    calculates the matching score between an id and a bb
    id is a dict:
        'bb': np.array([det[2:6]])
        'v_list': np.zeros((6, 2), dtype='float32')
        'p_list': np.zeros((6, 1), dtype='float32')
        'max_score': det[-1]
        'f_start': f_num
        'pred': int num
    bb: np.array(det[2:6])
    returns a matching score with type float
    """

    v_loss = v_model.evaluate(
        x=np.array([id_['v_list']]), y=get_v(id_['bb'][-1], bb, resolution),
        batch_size=1, verbose=0)
    p_loss = p_model.evaluate(
        x=np.array([id_['p_list']]), y=get_iou(id_['bb'][-1], bb),
        batch_size=1, verbose=0)
    v_loss, p_loss = v_loss[0], p_loss[0]
    return ds_model.predict(x=np.array([[v_loss, p_loss]], dtype='float32'),
                            batch_size=1, verbose=0)


def bb_update_vp(id_, bb, resolution):
    """
    updates v_list and p_list for id_
    the format of input parameters is the same as ds_score
    returns nothing
    """
    id_['v_list'] = np.delete(id_['v_list'], (0), axis=0)    
    id_['v_list'] = np.append(id_['v_list'], get_v(
        id_['bb'][-1], bb, resolution), axis=0)
    id_['p_list'] = np.delete(id_['p_list'], (0), axis=0)
    id_['p_list'] = np.append(id_['p_list'], get_iou(
        id_['bb'][-1], bb), axis=0)


def bb_pred(id_, resolution):
    """
    predicts next bb for id_
    the format of input parameters is the same as ds_score
    returns nothing
    """
    v = v_model.predict(x=np.array([id_['v_list']]), batch_size=1, verbose=0)
    new_bb = id_['bb'][-1]
    new_bb[0] += v[0][0] * resolution[0] / scale_factor
    new_bb[1] += v[0][1] * resolution[1] / scale_factor
    bb_update_vp(id_, new_bb, resolution)
    id_['bb'].append(new_bb)
    id_['pred'] += 1


def bb_update_vp2(id_, bb, resolution):
    """
    updates v_list and p_list for an id_ with short length
    the format of input parameters is the same as ds_score
    returns nothing
    """
    id_['v_list'][len(id_['bb']) - 1] = get_v(id_['bb'][-1], bb, resolution)
    id_['p_list'][len(id_['bb']) - 1] = get_iou(id_['bb'][-1], bb)
