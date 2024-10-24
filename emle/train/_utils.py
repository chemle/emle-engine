import torch as _torch


def pad_to_shape(tensor, shape, value=0):
    """Pads the input tensor to the specified shape with a constant value."""
    pad = [(0, n_max - n) for n_max, n in zip(shape, tensor.shape)]
    padding = [item for sublist in pad for item in sublist]
    return _torch.nn.functional.pad(tensor, padding, value=value)

def pad_to_max(arrays, value=0):
    """Pads tensors in the list to the size of the largest tensor along each axis."""
    shape = _torch.max(_torch.tensor([tensor.shape for tensor in arrays]), dim=0)[0]
    padded_arrays = [pad_to_shape(tensor, shape, value) for tensor in arrays]
    
    return padded_arrays

def mean_by_z(arr, zid):
    max_index = _torch.max(zid).item()
    mean_values = _torch.tensor([_torch.mean(arr[zid == i]) for i in range(max_index + 1)])
    return mean_values