__author__ = 'Brian Kenji Iwana'

import numpy as np
import utils.dtw as dtw
from tqdm import tqdm

TRUNCATE_PRE = "TRUNCATE_PRE"
TRUNCATE_OUTER = "TRUNCATE_OUTER"
TRUNCATE_POST = "TRUNCATE_POST"
ZERO_PAD_PRE = "ZERO_PAD_PRE"
ZERO_PAD_OUTER = "ZERO_PAD_OUTER"
ZERO_PAD_POST = "ZERO_PAD_POST" # default
ZERO_PAD_MID = "ZERO_PAD_MID"
INTERPOLATE = "INTERPOLATE"
EDGE_PAD_PRE = "EDGE_PAD_PRE"
EDGE_PAD_OUTER = "EDGE_PAD_OUTER"
EDGE_PAD_POST = "EDGE_PAD_POST"
NOISE_PAD_PRE = "NOISE_PAD_PRE" #https://arxiv.org/pdf/1910.04341.pdf
NOISE_PAD_OUTER = "NOISE_PAD_OUTER" #https://arxiv.org/pdf/1910.04341.pdf
NOISE_PAD_POST = "NOISE_PAD_POST" #https://arxiv.org/pdf/1910.04341.pdf
STRF_PAD = "STRF_PAD" # https://www.nature.com/articles/s41598-020-71450-8
RANDOM_PAD = "RANDOM_PAD" # https://www.nature.com/articles/s41598-020-71450-8
ZOOM_PAD = "ZOOM_PAD" # https://www.nature.com/articles/s41598-020-71450-8
NEAREST_GUIDED_WARPING_A = "NEAREST_GUIDED_WARPING_A"
NEAREST_GUIDED_WARPING_A_CW = "NEAREST_GUIDED_WARPING_A_CW"
NEAREST_GUIDED_WARPING_AB = "NEAREST_GUIDED_WARPING_AB"
NEAREST_GUIDED_WARPING_AB_CW = "NEAREST_GUIDED_WARPING_AB_CW"

def _get_lengths(x):
    input_shape = np.shape(x)
    maxlength = input_shape[1]
    train_lengths = np.zeros(input_shape[0])
    for t, x_t in enumerate(x):
        if np.isnan(x_t).any():
            nan_start = np.where(np.isnan(x_t))[0][0]
            num_nan = maxlength - nan_start
            train_lengths[t] = nan_start
        else:
            train_lengths[t] = maxlength
    return train_lengths

def truncate_pre(x_data_train, x_data_test):
    train_lengths = _get_lengths(x_data_train)
    minimum = int(np.min(train_lengths))
    arr_train = np.zeros((np.shape(x_data_train)[0], minimum))
    arr_test = np.zeros((np.shape(x_data_test)[0], minimum))
    for t, x_t in enumerate(x_data_train):
        curr = x_t[:int(train_lengths[t])]
        arr_train[t] = curr[-minimum:]
    for t, x_t in enumerate(x_data_test):
        if np.isnan(x_t).any():
            nan_start = np.where(np.isnan(x_t))[0][0]
            curr = x_t[:nan_start]
        else:
            curr = x_t
        length = len(curr)
        if length < minimum:
            extrazeros = minimum - length
            curr = np.append(np.zeros(extrazeros), curr)
        arr_test[t] = curr[-minimum:]
    return arr_train, arr_test

def truncate_outer(x_data_train, x_data_test):
    train_lengths = _get_lengths(x_data_train)
    minimum = int(np.min(train_lengths))
    arr_train = np.zeros((np.shape(x_data_train)[0], minimum))
    arr_test = np.zeros((np.shape(x_data_test)[0], minimum))
    for t, x_t in enumerate(x_data_train):
        curr = x_t[:int(train_lengths[t])]
        start = int(np.floor(train_lengths[t]/2.)) - int(np.floor(minimum/2.))
        stop = int(np.floor(train_lengths[t]/2.)) + int(np.ceil(minimum/2.))
        arr_train[t] = curr[start:stop]
    for t, x_t in enumerate(x_data_test):
        if np.isnan(x_t).any():
            nan_start = np.where(np.isnan(x_t))[0][0]
            curr = x_t[:nan_start]
        else:
            curr = x_t
        length = len(curr)
        if length < minimum:
            extrazeros = minimum - length
            curr = np.append(curr, np.zeros(extrazeros))
            arr_test[t] = curr
        else:
            start = int(np.floor(length/2.)) - int(np.floor(minimum/2.))
            stop = int(np.floor(length/2.)) + int(np.ceil(minimum/2.))
            arr_test[t] = curr[start:stop]
    return arr_train, arr_test

def truncate_post(x_data_train, x_data_test):
    train_lengths = _get_lengths(x_data_train)
    minimum = int(np.min(train_lengths))
    return x_data_train[:,:minimum], np.nan_to_num(x_data_test[:,:minimum])


def zero_pad_pre(x_data_train, x_data_test):
    return _zero_pad_pre(x_data_train), _zero_pad_pre(x_data_test)

def _zero_pad_pre(x_data):
    maxlength = np.shape(x_data)[1]
    for t, x_t in enumerate(x_data):
        if np.isnan(x_t).any():
            nan_start = np.where(np.isnan(x_t))[0][0]
            num_nan = maxlength - nan_start
            x_data[t] = np.append(np.zeros(num_nan), x_t[:nan_start])
    return x_data
                    
def zero_pad_outer(x_data_train, x_data_test):
    return _zero_pad_outer(x_data_train), _zero_pad_outer(x_data_test)

def _zero_pad_outer(x_data):
    maxlength = np.shape(x_data)[1]
    for t, x_t in enumerate(x_data):
        if np.isnan(x_t).any():
            nan_start = np.where(np.isnan(x_t))[0][0]
            num_nan = maxlength - nan_start
            
            pad_pre = np.array(np.zeros(int(np.floor(num_nan/2.))))
            pad_post = np.array(np.zeros(int(np.ceil(num_nan/2.))))
            padded = np.append(pad_pre, x_t[:nan_start])
            x_data[t] = np.append(padded, pad_post)
    return x_data

def zero_pad_mid(x_data_train, x_data_test):
    return _zero_pad_mid(x_data_train), _zero_pad_mid(x_data_test)

def _zero_pad_mid(x_data):
    maxlength = np.shape(x_data)[1]
    for t, x_t in enumerate(x_data):
        if np.isnan(x_t).any():
            nan_start = np.where(np.isnan(x_t))[0][0]
            num_nan = maxlength - nan_start
            
            
            pre = x_t[:int(nan_start/2.)]
            post = x_t[int(nan_start/2.):nan_start]
            padded = np.append(pre, np.zeros(num_nan))
            x_data[t] = np.append(padded, post)
    return x_data

def zero_pad_post(x_data_train, x_data_test):
    return np.nan_to_num(x_data_train), np.nan_to_num(x_data_test)

def noise_pad_pre(x_data_train, x_data_test):
    return _noise_pad_pre(x_data_train), _noise_pad_pre(x_data_test)

def _noise_pad_pre(x_data):
    maxlength = np.shape(x_data)[1]
    for t, x_t in enumerate(x_data):
        if np.isnan(x_t).any():
            nan_start = np.where(np.isnan(x_t))[0][0]
            num_nan = maxlength - nan_start
            x_data[t] = np.append(np.random.uniform(0, .001, num_nan), x_t[:nan_start])
    return x_data
                    
def noise_pad_outer(x_data_train, x_data_test):
    return _noise_pad_outer(x_data_train), _noise_pad_outer(x_data_test)

def _noise_pad_outer(x_data):
    maxlength = np.shape(x_data)[1]
    for t, x_t in enumerate(x_data):
        if np.isnan(x_t).any():
            nan_start = np.where(np.isnan(x_t))[0][0]
            num_nan = maxlength - nan_start
            
            pad_pre = np.array(np.random.uniform(0, .001, int(np.floor(num_nan/2.))))
            pad_post = np.array(np.random.uniform(0, .001, int(np.ceil(num_nan/2.))))
            padded = np.append(pad_pre, x_t[:nan_start])
            x_data[t] = np.append(padded, pad_post)
    return x_data

def noise_pad_post(x_data_train, x_data_test):
    return _noise_pad_post(x_data_train), _noise_pad_post(x_data_test)

def _noise_pad_post(x_data):
    maxlength = np.shape(x_data)[1]
    for t, x_t in enumerate(x_data):
        if np.isnan(x_t).any():
            nan_start = np.where(np.isnan(x_t))[0][0]
            num_nan = maxlength - nan_start
            x_data[t] = np.append(x_t[:nan_start], np.random.uniform(0, .001, num_nan))
    return x_data

def edge_pad_pre(x_data_train, x_data_test):
    return _edge_pad_pre(x_data_train), _edge_pad_pre(x_data_test)

def _edge_pad_pre(x_data):
    maxlength = np.shape(x_data)[1]
    for t, x_t in enumerate(x_data):
        if np.isnan(x_t).any():
            nan_start = np.where(np.isnan(x_t))[0][0]
            num_nan = maxlength - nan_start
            
            x_data[t] = np.append([x_t[0] for i in range(num_nan)], x_t[:nan_start])
    return x_data

def edge_pad_outer(x_data_train, x_data_test):
    return _edge_pad_outer(x_data_train), _edge_pad_outer(x_data_test)

def _edge_pad_outer(x_data):
    maxlength = np.shape(x_data)[1]
    for t, x_t in enumerate(x_data):
        if np.isnan(x_t).any():
            nan_start = np.where(np.isnan(x_t))[0][0]
            num_nan = maxlength - nan_start
            
            pad_pre = np.array([x_t[0] for i in range(int(np.floor(num_nan/2.)))])
            pad_post = np.array([x_t[nan_start-1] for i in range(int(np.ceil(num_nan/2.)))])
            padded = np.append(pad_pre, x_t[:nan_start])
            x_data[t] = np.append(padded, pad_post)
    return x_data

def edge_pad_post(x_data_train, x_data_test):
    return _edge_pad_post(x_data_train), _edge_pad_post(x_data_test)

def _edge_pad_post(x_data):
    maxlength = np.shape(x_data)[1]
    for t, x_t in enumerate(x_data):
        if np.isnan(x_t).any():
            nan_start = np.where(np.isnan(x_t))[0][0]
            num_nan = maxlength - nan_start
            
            x_data[t] = np.nan_to_num(x_t, nan=x_t[nan_start-1])
    return x_data

def interpolate(x_data_train, x_data_test):
    return _interpolate(x_data_train), _interpolate(x_data_test)

def _interpolate(x_data):
    maxlength = np.shape(x_data)[1]
    for t, x_t in enumerate(x_data):
        if np.isnan(x_t).any():
            nan_start = np.where(np.isnan(x_t))[0][0]
            num_nan = maxlength - nan_start
            
            x_data[t] = np.interp(np.linspace(0, nan_start-1, num=maxlength), np.arange(nan_start), x_t[:nan_start])
    return x_data

def strf_pad(x_data_train, x_data_test):
    return _strf_pad(x_data_train), _strf_pad(x_data_test)

def _strf_pad(x_data):
    maxlength = np.shape(x_data)[1]
    for t, x_t in enumerate(x_data):
        if np.isnan(x_t).any():
            nan_start = np.where(np.isnan(x_t))[0][0]
            num_nan = maxlength - nan_start
            
            x_data[t] = np.insert(x_t[:nan_start], np.linspace(1, nan_start-1, num=num_nan).astype(int), 0)
    return x_data

def random_pad(x_data_train, x_data_test):
    return _random_pad(x_data_train), _random_pad(x_data_test)

def _random_pad(x_data):
    maxlength = np.shape(x_data)[1]
    for t, x_t in enumerate(x_data):
        if np.isnan(x_t).any():
            nan_start = np.where(np.isnan(x_t))[0][0]
            num_nan = maxlength - nan_start
            
            if nan_start > 2:
                x_data[t] = np.insert(x_t[:nan_start], np.random.randint(1, nan_start-1, size=num_nan), 0)
            else:
                x_data[t] = np.insert(x_t[:nan_start], np.ones(num_nan).astype(int), 0)
    return x_data

def zoom_pad(x_data_train, x_data_test):
    return _zoom_pad(x_data_train), _zoom_pad(x_data_test)

def _zoom_pad(x_data):
    maxlength = np.shape(x_data)[1]
    for t, x_t in enumerate(x_data):
        if np.isnan(x_t).any():
            nan_start = np.where(np.isnan(x_t))[0][0]
            num_nan = maxlength - nan_start
            
            r_t = x_t[:nan_start]
            x_data[t] = r_t[np.linspace(0, nan_start-1, num=maxlength).astype(int)]
    return x_data

def nearest_guided_warping_zero(x_data_train, x_data_test, beta=None, slope_constraint="asymmetric"):
    input_shape = np.shape(x_data_train)
    maxlength = input_shape[1]
    train_lengths = _get_lengths(x_data_train)
    test_lengths = _get_lengths(x_data_test)
    
    if beta:
        upper_quantile = beta
    else:
        upper_quantile = 0.5
    print("Upper Quantile", upper_quantile)
    u_quant = int(np.quantile(train_lengths, upper_quantile))
    print("Upper Quantile Length", u_quant)
        
    prototype_ids=np.arange(input_shape[0])
    prototypes = np.zeros((len(prototype_ids), u_quant))
    test_ids=np.arange(len(x_data_test))
    
    ret_x_train = np.zeros((input_shape[0], u_quant))
    ret_x_test = np.zeros((np.shape(x_data_test)[0], u_quant))
    
    for t, p_t in enumerate(prototype_ids):
        nan_start = int(train_lengths[p_t])
        prot = x_data_train[p_t][:nan_start]
        prototypes[t] = np.interp(np.linspace(0, nan_start-1, num=u_quant), np.arange(nan_start), prot)
        ret_x_train[p_t] = prototypes[t]
        
    ret_x_test = _nearest_guided_warping(x_data_test, ret_x_test, test_lengths, prototypes, test_ids, slope_constraint=slope_constraint)
    return ret_x_train, ret_x_test

def nearest_guided_warping(x_data_train, x_data_test, alpha=0.8, beta=None, slope_constraint="asymmetric"):
    input_shape = np.shape(x_data_train)
    maxlength = input_shape[1]
    train_lengths = _get_lengths(x_data_train)
    test_lengths = _get_lengths(x_data_test)
    
    lower_quantile = alpha
    if beta:
        upper_quantile = beta
    else:
        upper_quantile = 1. - ( 1. - alpha ) / 2.
    print("Lower Quantile", lower_quantile)
    print("Upper Quantile", upper_quantile)
    l_quant = int(np.quantile(train_lengths, lower_quantile))
    u_quant = int(np.quantile(train_lengths, upper_quantile))
    print("Lower Quantile Length", l_quant)
    print("Upper Quantile Length", u_quant)
        
    prototype_ids = np.where(train_lengths >= l_quant)[0]
    prototypes = np.zeros((len(prototype_ids), u_quant))
    other_ids = np.where(train_lengths < l_quant)[0]
    test_ids=np.arange(len(x_data_test))
    
    ret_x_train = np.zeros((input_shape[0], u_quant))
    ret_x_test = np.zeros((np.shape(x_data_test)[0], u_quant))
    
    for t, p_t in enumerate(prototype_ids):
        nan_start = int(train_lengths[p_t])
        prot = x_data_train[p_t][:nan_start]
        prototypes[t] = np.interp(np.linspace(0, nan_start-1, num=u_quant), np.arange(nan_start), prot)
        ret_x_train[p_t] = prototypes[t]
        
    ret_x_train = _nearest_guided_warping(x_data_train, ret_x_train, train_lengths, prototypes, other_ids, slope_constraint=slope_constraint)
    ret_x_test = _nearest_guided_warping(x_data_test, ret_x_test, test_lengths, prototypes, test_ids, slope_constraint=slope_constraint)
    return ret_x_train, ret_x_test

def _nearest_guided_warping(x, ret_x, nan_starts, prototypes, ids, slope_constraint="asymmetric"):
    for t, s_t in enumerate(tqdm(ids)):
        dtw_dists = np.zeros(np.shape(prototypes)[0])
        nan_start = int(nan_starts[s_t])
        sample = x[s_t][:nan_start]
        if nan_start > 2.*np.shape(prototypes)[1]:
            ret_x[s_t] = np.interp(np.linspace(0, nan_start-1, num=np.shape(prototypes)[1]), np.arange(nan_start), sample)
            print('interpolating long sample')
        else:
            for i, p_i in enumerate(prototypes):
                dtw_dists[i] = dtw.dtw(p_i.reshape((-1, 1)), sample.reshape((-1, 1)), dtw.RETURN_VALUE, slope_constraint=slope_constraint)
            smallest_p = np.argmin(dtw_dists)
            path = dtw.dtw(prototypes[smallest_p].reshape((-1, 1)), sample.reshape((-1, 1)), dtw.RETURN_PATH, slope_constraint=slope_constraint)
            ret_x[s_t] = sample[path[1]]
    return ret_x

def nearest_guided_warping_classwise(x_data_train, x_data_test, alpha=0.8, beta=None, slope_constraint="asymmetric", y_labels_train=None, nb_class=None):
    
    input_shape = np.shape(x_data_train)
    maxlength = input_shape[1]
    train_lengths = _get_lengths(x_data_train)
    test_lengths = _get_lengths(x_data_test)
    
    lower_quantile = alpha
    if beta:
        upper_quantile = beta
    else:
        upper_quantile = 1. - ( 1. - alpha ) / 2.
    print("Lower Quantile", lower_quantile)
    print("Upper Quantile", upper_quantile)
    l_quant = int(np.quantile(train_lengths, lower_quantile))
    u_quant = int(np.quantile(train_lengths, upper_quantile))
    print("Lower Quantile Length", l_quant)
    print("Upper Quantile Length", u_quant)
    
    prototype_ids = []
    other_ids = []
    for c in range(nb_class):
        class_train_ids = np.where(y_labels_train == c)[0]
        prototype_class_train_ids = np.where(train_lengths[class_train_ids] >= l_quant)[0]
        class_prototype_ids = class_train_ids[prototype_class_train_ids]
        
        other_class_train_ids = np.where(train_lengths[class_train_ids] < l_quant)[0]
        class_other_ids = class_train_ids[other_class_train_ids]
        
        prototype_ids = np.append(prototype_ids, class_prototype_ids)
        other_ids = np.append(other_ids, class_other_ids)
    
    prototype_ids = prototype_ids.astype(int)
    other_ids = other_ids.astype(int)
    prototypes = np.zeros((len(prototype_ids), u_quant))
    test_ids=np.arange(len(x_data_test))
    
    ret_x_train = np.zeros((input_shape[0], u_quant))
    ret_x_test = np.zeros((np.shape(x_data_test)[0], u_quant))
    
    for t, p_t in enumerate(prototype_ids):
        nan_start = int(train_lengths[p_t])
        prot = x_data_train[p_t][:nan_start]
        prototypes[t] = np.interp(np.linspace(0, nan_start-1, num=u_quant), np.arange(nan_start), prot)
        ret_x_train[p_t] = prototypes[t]
        
    ret_x_train = _nearest_guided_warping(x_data_train, ret_x_train, train_lengths, prototypes, other_ids, slope_constraint=slope_constraint)
    ret_x_test = _nearest_guided_warping(x_data_test, ret_x_test, test_lengths, prototypes, test_ids, slope_constraint=slope_constraint)
    return ret_x_train, ret_x_test

def fix_length(x_train, x_test, fix=ZERO_PAD_POST):
    if fix == TRUNCATE_PRE:
        return truncate_pre(x_train, x_test)
    elif fix == TRUNCATE_OUTER:
        return truncate_outer(x_train, x_test)
    elif fix == TRUNCATE_POST:
        return truncate_post(x_train, x_test)
    elif fix == ZERO_PAD_PRE:
        return zero_pad_pre(x_train, x_test)
    elif fix == ZERO_PAD_OUTER:
        return zero_pad_outer(x_train, x_test)
    elif fix == ZERO_PAD_POST:
        return zero_pad_post(x_train, x_test)
    elif fix == ZERO_PAD_MID:
        return zero_pad_mid(x_train, x_test)
    elif fix == INTERPOLATE:
        return interpolate(x_train, x_test)
    elif fix == EDGE_PAD_PRE:
        return edge_pad_pre(x_train, x_test)
    elif fix == EDGE_PAD_OUTER:
        return edge_pad_outer(x_train, x_test)
    elif fix == EDGE_PAD_POST:
        return edge_pad_post(x_train, x_test)
    elif fix == NOISE_PAD_PRE:
        return noise_pad_pre(x_train, x_test)
    elif fix == NOISE_PAD_OUTER:
        return noise_pad_outer(x_train, x_test)
    elif fix == NOISE_PAD_POST:
        return noise_pad_post(x_train, x_test)
    elif fix == STRF_PAD:
        return strf_pad(x_train, x_test)
    elif fix == RANDOM_PAD:
        return random_pad(x_train, x_test)
    elif fix == ZOOM_PAD:
        return zoom_pad(x_train, x_test)
    elif fix == NEAREST_GUIDED_WARPING_A:
        return nearest_guided_warping(x_train, x_test, alpha=0.4, beta=1.0)
    elif fix == NEAREST_GUIDED_WARPING_AB:
        return nearest_guided_warping(x_train, x_test, alpha=0.4, beta=0.7)
    elif fix == NEAREST_GUIDED_WARPING_A_CW:
        print("please run the classwise version of NGW directly)
    elif fix == NEAREST_GUIDED_WARPING_AB_CW:
        print("please run the classwise version of NGW directly)
    else:
        print("normalization missing")
    return model
