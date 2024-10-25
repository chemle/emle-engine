import torch as _torch
import numpy as _np


def pad_to_shape(tensor, shape, value=0):
    """Pads the input tensor to the specified shape with a constant value."""
    pad = [(0, n_max - n) for n_max, n in zip(shape, tensor.shape)]
    padding = [item for sublist in pad for item in sublist]
    return _torch.nn.functional.pad(tensor, padding, value=value)

def pad_to_max(arrays, value=0):
    """
    Pad tensors in the list/array/tensor to the size of the largest tensor along each axis.
    
    Parameters
    ----------
    arrays : iterable of torch.Tensor or np.ndarray or list
        Iterable of data to be padded.
    value : float, optional
        Value to pad with.

    Returns
    -------
    list of torch.Tensor or np.ndarray or list
        Padded data.
    """
    # Convert to torch.Tensor
    tensors = [_torch.from_numpy(array) if isinstance(array, _np.ndarray) else array for array in arrays]
    padded_arrays = _torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True)
    return padded_arrays

def mean_by_z(arr, zid):
    """
    Calculate the mean of the input array by the zid.

    Parameters
    ----------
    arr : torch.Tensor(N_BATCH, MAX_N_ATOMS)
        Input array.
    zid : torch.Tensor(N_BATCH, MAX_N_ATOMS)
        Species indices.

    Returns
    -------
    torch.Tensor(N_SPECIES)
        Mean values by species.
    """
    max_index = _torch.max(zid).item()
    mean_values = _torch.tensor([_torch.mean(arr[zid == i]) for i in range(max_index + 1)])
    return mean_values