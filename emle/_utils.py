import numpy as _np

def pad_to_shape(array, shape, value=0):
    pad = [(0, n_max - n) for n_max, n in zip(shape, array.shape)]
    return _np.pad(array, pad, constant_values=value)


def pad_to_max(arrays, value=0):
    # Takes arrays with different shapes, but same number of dimensions
    # and pads them to the size of the largest array along each axis
    shape = _np.max([_.shape for _ in arrays], axis=0)
    return _np.array([pad_to_shape(_, shape, value) for _ in arrays])