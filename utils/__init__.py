from pathlib import Path
from typing import Iterable

import torch
import torch.distributed

from utils import parallel


def send_to_device(batch_data):
    if isinstance(batch_data, dict):
        new_batch_data = {}
        for key, data in batch_data.items():
            new_batch_data[key] = send_to_device(data)
    elif isinstance(batch_data, Iterable) and not isinstance(batch_data, torch.Tensor):
        new_batch_data = (send_to_device(data) for data in batch_data)
    elif isinstance(batch_data, torch.Tensor):
        new_batch_data = batch_data.cuda()
    else:
        new_batch_data = batch_data
    return new_batch_data


def reduce_all_iterable(iterable):
    list_to_send = []

    def _value_to_getter(iterable):
        if isinstance(iterable, dict):
            return dict((key, _value_to_getter(value)) for key, value in iterable.items())

        if isinstance(iterable, list):
            return [_value_to_getter(value) for value in iterable]

        if isinstance(iterable, tuple):
            return tuple(_value_to_getter(value) for value in iterable)

        if isinstance(iterable, set):
            return set(_value_to_getter(value) for value in iterable)

        if isinstance(iterable, int) or isinstance(iterable, float) or isinstance(iterable, bool):
            list_to_send.append(iterable)
            idx = len(list_to_send) - 1
            t = type(iterable)
            if t == bool:
                def t(x):
                    return bool(int(round(x)))

                return idx, lambda li, idx: t(li[idx])
            else:
                return idx, lambda li, idx: li[idx]

        return iterable

    def _getter_to_value(li, getter_iterable):
        if isinstance(getter_iterable, tuple) and \
                len(getter_iterable) == 2 and \
                callable(getter_iterable[1]):
            return getter_iterable[1](li, getter_iterable[0])

        if isinstance(getter_iterable, dict):
            return dict((key, _getter_to_value(li, value)) for key, value in getter_iterable.items())

        if isinstance(getter_iterable, list):
            return [_getter_to_value(li, value) for value in getter_iterable]

        if isinstance(getter_iterable, tuple):
            return tuple(_getter_to_value(li, value) for value in getter_iterable)

        if isinstance(getter_iterable, set):
            return set(_getter_to_value(li, value) for value in getter_iterable)

        return getter_iterable

    getter_iterable = _value_to_getter(iterable)
    tensor_to_reduce = torch.tensor(list_to_send, dtype=torch.float).cuda()
    torch.distributed.all_reduce(tensor_to_reduce)
    list_to_send = (tensor_to_reduce / int(parallel.get_size())).tolist()
    iterable = _getter_to_value(list_to_send, getter_iterable)

    return iterable


def save_models(checkpoint_path, networks, criterions=None, optimizers=None, **kwargs):
    if criterions is None:
        criterions = dict()

    if optimizers is None:
        optimizers = dict()

    torch.save({
        'networks': dict((key, module.state_dict()) for key, module in networks.items()),
        'criterions': dict((key, criterion.state_dict()) for key, criterion in criterions.items()),
        'optimizers': optimizers,
        **kwargs
    }, Path(checkpoint_path))


def load_models(checkpoint_path, networks, criterions=None, optimizers=None, kwargs=None, map_location=None,
                remove_parallel_key=False, strict=True):
    if map_location is None:
        map_location = str(torch.cuda.current_device())
    checkpoint = torch.load(Path(checkpoint_path).as_posix(), map_location=map_location)

    for name in networks:
        if remove_parallel_key:
            model_state_dict = {}
            for k, v in checkpoint['networks'][name].items():
                model_state_dict[k[7:]] = v  # remove 'module.' from key
        else:
            model_state_dict = checkpoint['networks'][name]

        networks[name].load_state_dict(model_state_dict, strict)
        networks[name].to(map_location)

    if criterions is not None:
        for name in criterions:
            if remove_parallel_key:
                model_state_dict = {}
                for k, v in checkpoint['criterions'][name].items():
                    model_state_dict[k[7:]] = v  # remove 'module.' from key
            else:
                model_state_dict = checkpoint['criterions'][name]
            criterions[name].load_state_dict(model_state_dict, strict)
            criterions[name].to(map_location)

    if optimizers is not None:
        for name in optimizers:
            optimizers[name].load_state_dict(checkpoint['optimizers'][name])

    if kwargs is not None:
        for name in kwargs:
            kwargs[name] = checkpoint[name]
