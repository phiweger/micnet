import numpy as np


def distance(ref: np.array, qry: np.array, metric='euclidean'):
    '''
    a = np.array([np.log(i) for i in [1, 0.125, 16, 0.03125, 256, 0.5, 0.5]])
    b = np.array([np.log(i) for i in [1, 0.125, 32, 0.03125, 256, 0.5, 0.5]])

    calc_distance(a, b)
    '''
    if metric == 'euclidean':
        return np.linalg.norm(ref - qry)
    else:
        raise ValueError('Not implemented')
