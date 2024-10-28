import torch as _torch
import numpy as _np


def pad_to_max(arrays, value=0, side="right"):
    """
    Pad tensors in the list/array/tensor to the size of the largest tensor along each axis.

    Parameters
    ----------
    arrays : iterable of torch.Tensor or np.ndarray or list
        Iterable of data to be padded.
    value : float, optional, default=0
        Value to pad with.
    side : str, optional, default='right'
        The side to pad the sequences on.

    Returns
    -------
    list of torch.Tensor or np.ndarray or list
        Padded data.
    """
    # Convert to torch.Tensor
    tensors = [
        _torch.from_numpy(array) if isinstance(array, _np.ndarray) else array
        for array in arrays
    ]
    padded_arrays = _torch.nn.utils.rnn.pad_sequence(
        tensors, batch_first=True, padding_value=value, padding_side=side
    )
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
    mean_values = _torch.tensor(
        [_torch.mean(arr[zid == i]) for i in range(max_index + 1)]
    )
    return mean_values
