import numpy as np
from typing import List, Dict, Any, Union
import torch

# from gymnasium.wrappers.frame_stack import LazyFrames
from omegaconf import DictConfig


def auto_stack(elems: List[Any]) -> Any:
    if isinstance(elems[0], np.ndarray):
        return np.stack(elems)
    elif isinstance(elems[0], torch.Tensor):
        return torch.stack(elems)
    elif isinstance(elems[0], str):
        return elems
    elif isinstance(elems[0], dict) or isinstance(elems[0], DictConfig):
        ret = {}
        for key in elems[0].keys():
            ret[key] = auto_stack([d[key] for d in elems])
        return ret
    elif isinstance(elems[0], list):
        return [auto_stack([l[i] for l in elems]) for i in range(len(elems[0]))]
    elif isinstance(elems[0], (int, float, bool, np.number)):
        return np.array(elems)
    # elif isinstance(elems[0], LazyFrames):
    #     return np.array(elems)
    else:
        return elems


def auto_to_numpy(data: Any) -> Any:
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    elif isinstance(data, dict):
        ret = {}
        for key, value in data.items():
            ret[key] = auto_to_numpy(value)
        return ret
    elif isinstance(data, list):
        return [auto_to_numpy(d) for d in data]
    else:
        return data


def auto_to_torch(data: Any, device: Union[str, torch.device]) -> Any:
    if isinstance(data, np.ndarray):
        np_data = torch.tensor(data)
        return np_data.to(device)
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        ret = {}
        for key, value in data.items():
            ret[key] = auto_to_torch(value, device)
        return ret
    elif isinstance(data, list):
        return [auto_to_torch(d, device) for d in data]
    elif isinstance(data, tuple):
        return tuple([auto_to_torch(d, device) for d in data])
    else:
        return data


def auto_getitem(
    data: Union[Dict[str, torch.Tensor], torch.Tensor], index: int
) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
    if isinstance(data, dict):
        return {key: value[index] for key, value in data.items()}
    elif isinstance(data, torch.Tensor):
        return data[index]
    else:
        raise NotImplementedError


def auto_slice(data, start, end, dim, type_list=0):
    if data is None:
        return None
    if isinstance(data, torch.Tensor):
        return data.narrow(dim, start, end - start)
    elif isinstance(data, dict) or isinstance(data, DictConfig):
        return {k: auto_slice(v, start, end, dim, type_list) for k, v in data.items()}
    elif isinstance(data, list):
        if dim == 0 or type_list == 1:
            return data[start:end]
        else:
            return [auto_slice(v, start, end, dim - 1, type_list) for v in data]
    else:
        raise ValueError(f"Unsupported data type {type(data)}")


def auto_cat(elems: List[Any], dim: int = 0) -> Any:
    if elems[0] is None:
        return None
    elif isinstance(elems[0], np.ndarray):
        return np.concatenate(elems, axis=dim)
    elif isinstance(elems[0], torch.Tensor):
        return torch.cat(elems, dim=dim)
    elif isinstance(elems[0], list):
        if dim == 0:
            return sum(elems, [])
        else:
            return [auto_cat([l[i] for l in elems], dim - 1) for i in range(len(elems[0]))]
    elif isinstance(elems[0], dict):
        ret = {}
        for key in elems[0].keys():
            ret[key] = auto_cat([d[key] for d in elems], dim=dim)
        return ret
    else:
        raise ValueError(f"Unsupported data type {type(elems[0])}")


def auto_pad(arr: Any, padding_length: int) -> Any:
    if isinstance(arr, np.ndarray):
        return np.pad(arr, ((0, padding_length),) + ((0, 0),) * (len(arr.shape) - 1), mode="constant")
    elif isinstance(arr, list):
        return arr + [arr[0] for _ in range(padding_length)]
    elif isinstance(arr, dict):
        ret = {}
        for key, value in arr.items():
            ret[key] = auto_pad(value, padding_length)
        return ret
    else:
        raise NotImplementedError


def recursive_detach(data):
    if isinstance(data, torch.Tensor):
        return data.detach()
    elif isinstance(data, list):
        return [recursive_detach(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(recursive_detach(item) for item in data)
    else:
        return data
