import datetime
import os
import platform
import random
import types
from typing import Callable, Dict, Any

import numpy as np
import torch
import torch.backends.cudnn
import torch.cuda.amp
import torch.distributed as dist
import torch.utils.data
from torch import nn
from torch.nn.parallel import DataParallel

try:
    from apex.parallel import DistributedDataParallel as ApexDDP, convert_syncbn_model
    from apex import amp

    __apex_available__ = True
except ModuleNotFoundError:
    __apex_available__ = False

    from torch.nn.parallel import DistributedDataParallel as TorchDDP
    from torch.nn import SyncBatchNorm

    convert_syncbn_model = SyncBatchNorm.convert_sync_batchnorm


def setup_multiprocess(init_method="env://", timeout=datetime.timedelta(seconds=1800),
                       store=None, group_name='', random_seed=0, deterministic=False):
    if 'DEVICE_ID' not in os.environ:
        raise ValueError('"DEVICE_ID must be included in environment variables')

    dist.init_process_group(backend='nccl',
                            world_size=get_size(), rank=get_rank(),
                            init_method=init_method,
                            timeout=timeout, store=store, group_name=group_name)

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic

    torch.cuda.set_device(torch.device(get_device_id()))


def cleanup_multiprocess():
    """ A function to clean up multi-processing """
    dist.destroy_process_group()


def is_distributed_available():
    return platform.system() != 'Windows' and torch.distributed.is_available()


def is_apex_available():
    return __apex_available__


def is_distributed_initialized():
    return platform.system() != 'Windows' and torch.distributed.is_initialized()


def is_first_process():
    return not is_distributed_initialized() or get_rank() == 0


def get_rank():
    return int(os.environ['RANK'])


def get_size():
    return int(os.environ['WORLD_SIZE'])


def get_local_rank():
    return int(os.environ['LOCAL_RANK'])


def get_device_id():
    return int(os.environ['DEVICE_ID'])


def wait_for_all_processes():
    dist.barrier()


def wrap_module_with_parallel(module, multiprocess=False, custom_wrapper: nn.Module = None,
                              to_syncbn_mode: Callable = None, dp_device_ids=None, **kwargs_to_wrapper):
    device = torch.cuda.current_device()

    if custom_wrapper is not None:
        wrapper = custom_wrapper
    elif multiprocess:
        if is_apex_available():
            wrapper = ApexDDP
        else:
            if 'device_ids' not in kwargs_to_wrapper:
                kwargs_to_wrapper['device_ids'] = [device]
            if 'output_device' not in kwargs_to_wrapper:
                kwargs_to_wrapper['output_device'] = device
            wrapper = TorchDDP
    else:
        if 'device_ids' not in kwargs_to_wrapper:
            kwargs_to_wrapper['device_ids'] = dp_device_ids
        if 'output_device' not in kwargs_to_wrapper:
            kwargs_to_wrapper['output_device'] = device
        wrapper = DataParallel

    if wrapper is not DataParallel:
        if to_syncbn_mode is None:
            module = convert_syncbn_model(module)
        else:
            module = to_syncbn_mode(module)
    return wrapper(module.to(device), **kwargs_to_wrapper)


def wrap_model_with_apex_amp(model, opt_level: str = 'O1', **kwargs_to_initializer):
    # initialize networks and optimizers
    networks = model.networks
    optimizers = model.optimizers

    network_names = networks.keys()
    optimizer_names = optimizers.keys()
    network_values = [networks[name] for name in network_names]
    optimizer_values = [optimizers[name] for name in optimizer_names]

    amp_networks, amp_optimizers = amp.initialize(network_values, optimizer_values, opt_level=opt_level,
                                                  **kwargs_to_initializer)

    for idx, name in enumerate(network_names):
        networks[name] = amp_networks[idx]

    for idx, name in enumerate(optimizer_names):
        optimizers[name] = amp_optimizers[idx]

    # add amp to save and load functions
    # create_checkpoint_contents
    previous_create_checkpoint_contents = model.create_checkpoint_contents

    def create_checkpoint_contents(self, object_names_to_save=None):
        prev = previous_create_checkpoint_contents(object_names_to_save)
        prev['amp'] = amp.state_dict()
        return prev

    model.create_checkpoint_contents = types.MethodType(create_checkpoint_contents, model)

    # load
    previous_checkpoint_to_model = model.checkpoint_to_model

    def checkpoint_to_model(self, checkpoint: Dict[str, Dict[str, Any]], device, object_names_to_load=None,
                            remove_parallel_key: bool = False, strict: bool = True):
        previous_checkpoint_to_model(checkpoint, device, object_names_to_load, remove_parallel_key, strict)
        amp.load_state_dict(checkpoint['amp'])

    model.checkpoint_to_model = types.MethodType(checkpoint_to_model, model)
    return model


def __wrap_module_with_autocast(module: nn.Module, **kwargs_to_autocast):
    previous_forward = module.forward

    def forward(self: nn.Module, *args, **kwargs):
        with torch.cuda.amp.autocast(**kwargs_to_autocast):
            return previous_forward(*args, **kwargs)

    module.forward = types.MethodType(forward, module)

    for attribute in module.__dict__.values():
        try:
            if 'forward' in attribute.__dict__:
                __wrap_module_with_autocast(attribute, **kwargs_to_autocast)
        except AttributeError:
            pass


def wrap_model_with_torch_autocast(model, **kwargs_to_autocast):
    for network in model.networks.values():
        __wrap_module_with_autocast(network, **kwargs_to_autocast)
    return model
