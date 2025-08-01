# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
import json
import math
import os
from collections import defaultdict
from typing import Callable, Dict, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from safetensors.torch import load_file
from torch.distributed._tensor import Replicate, Shard

from ...utils import logging
from ..parallel_plan import SpecInfo


logger = logging.get_logger(__name__)


def parallel_load_safetensors(
    filepath: str, specific_param_name: list[str] = None, ignore_param_name: list[str] = None
):
    assert not (specific_param_name is not None and ignore_param_name is not None)

    dist.barrier()

    safetensors2param = {}
    index_file = os.path.join(filepath, "model.safetensors.index.json")
    if os.path.exists(index_file):
        index = json.load(open(index_file, "rb"))
        for param_name, filename in index["weight_map"].items():
            if specific_param_name is not None:
                if param_name not in specific_param_name:
                    continue
            elif ignore_param_name is not None:
                if param_name in ignore_param_name:
                    continue
            safetensors2param.setdefault(filename, []).append(param_name)
    else:
        # in this case, the model is small and we can load it all at once
        param_file = os.path.join(filepath, "model.safetensors")
        assert os.path.exists(param_file), f"Cannot find {param_file}"
        states = load_file(param_file)
        for param_name in states:
            safetensors2param.setdefault("model.safetensors", []).append(param_name)
        del states

    total_files = len(safetensors2param)
    ckpt_chunks = sorted(safetensors2param.keys())
    world_size = dist.get_world_size()
    size = int(math.ceil(total_files / world_size))
    ckpt_chunks = [ckpt_chunks[i * size : (i + 1) * size] for i in range(world_size)]

    shard_states = {}
    device = torch.cuda.current_device()
    for rank, files in enumerate(ckpt_chunks):
        if rank == dist.get_rank():
            for file in files:
                safetensors_file = os.path.join(filepath, file)
                states = load_file(safetensors_file, device=device)
                valid_states = {k: v for k, v in states.items() if k in safetensors2param[file]}
                shard_states.update(valid_states)
                del states
        else:
            for file in files:
                for param_name in safetensors2param[file]:
                    shard_states[param_name] = rank
    return shard_states


def parallel_init_fsdp_fn(
    module: torch.nn.Module,
    shard_states: Dict[str, torch.nn.Parameter],
    remove_standalone: bool = True,
    specific_param_name: list[str] = None,
    ignore_param_name: list[str] = None,
):
    """
    Initialize a module with sharded states in a parallel fashion using Fully Sharded Data Parallel (FSDP).

    Args:
        module (torch.nn.Module): The module to be initialized.
        shard_states (Dict[str, torch.nn.Parameter]): A dictionary containing sharded states.
        remove_standalone (bool, optional): If True, only consider shared states. Defaults to True.
        specific_param_name (list[str], optional): A list of specific parameter names to consider. Defaults to None.
        ignore_param_name (list[str], optional): A list of parameter names to ignore. Defaults to None.

    Returns:
        Callable[[torch.nn.Module], torch.nn.Module]: A function that initializes sub-modules of the given module.
    """
    assert not (specific_param_name is not None and ignore_param_name is not None)
    state2fqn = {}
    for name, state in itertools.chain(
        module.named_parameters(remove_duplicate=False), module.named_buffers(remove_duplicate=False)
    ):
        if specific_param_name is not None:
            if name not in specific_param_name:
                continue
        elif ignore_param_name is not None:
            if name in ignore_param_name:
                continue
        state2fqn.setdefault(state, []).append(name)

    shared = {s for s, names in state2fqn.items() if len(names) > 1} if remove_standalone else set(state2fqn.keys())

    materialized_states = {}

    def make_full_tensor(param: torch.Tensor, spec_info: SpecInfo):
        """
        Create a full tensor from a sharded tensor based on the given specification.

        Args:
            param (torch.Tensor): The sharded tensor.
            spec_info (SpecInfo): The specification information.

        Returns:
            torch.Tensor: The full tensor.
        """
        device = torch.cuda.current_device()
        if isinstance(spec_info.placement, Replicate):
            return torch.empty_like(param.data, device=device)
        else:
            assert isinstance(spec_info.placement, Shard)
            size = list(param.shape)
            size[spec_info.placement.dim] *= spec_info.ep_mesh.size()
            return torch.empty(size, dtype=param.dtype, device=device)

    def copy_to_local(param: torch.Tensor, full_data: torch.Tensor, spec_info: SpecInfo):
        """
        Copy data from a full tensor to a local sharded tensor based on the given specification.

        Args:
            param (torch.Tensor): The local sharded tensor.
            full_data (torch.Tensor): The full tensor.
            spec_info (SpecInfo): The specification information.
        """
        if isinstance(spec_info.placement, Replicate):
            param.data.copy_(full_data)
        else:
            assert isinstance(spec_info.placement, Shard)
            local_data = full_data.chunk(spec_info.ep_mesh.size(), dim=spec_info.placement.dim)[
                spec_info.ep_mesh.get_local_rank()
            ]
            param.data.copy_(local_data.contiguous())
        param.spec_info = spec_info

    @torch.no_grad()
    def create_and_sync_state(param_name, state, is_param):
        """
        Create and synchronize a state tensor across multiple devices.

        Args:
            param_name (str): The name of the parameter.
            state (torch.Tensor): The state tensor.
            is_param (bool): Whether the state is a parameter or a buffer.

        Returns:
            torch.Tensor: The synchronized state tensor.
        """
        device = torch.cuda.current_device()
        if is_param:
            param = torch.nn.Parameter(torch.empty_like(state.data, device=device), requires_grad=state.requires_grad)
        else:  # buffer
            param = torch.empty_like(state.data, device=device)
        if param_name not in shard_states:
            logger.warn(f"{param_name} not found in shard states, init it from random")
            assert is_param
            if dist.get_rank() == 0:
                initializer_range = (2.5 * max(state.shape)) ** -0.5
                size = list(state.size())
                if hasattr(state, "spec_info"):
                    shard = state.spec_info.placement
                    if isinstance(shard, Shard):
                        size[shard.dim] *= state.spec_info.ep_mesh.size()
                shard_states[param_name] = torch.nn.Parameter(
                    torch.randn(size, dtype=state.dtype, device=device, requires_grad=state.requires_grad)
                    * initializer_range
                )
            else:
                shard_states[param_name] = 0
        loaded = shard_states[param_name]
        if isinstance(loaded, (torch.nn.Parameter, torch.Tensor)):
            loaded = loaded.to(dtype=param.dtype, device=device)
            dist.broadcast(loaded.data.to(param.dtype), src=dist.get_rank())
            if hasattr(state, "spec_info"):
                copy_to_local(param, loaded.data, state.spec_info)
            else:
                param.data.copy_(loaded.data)
        else:
            assert isinstance(loaded, int)  # the rank that holds the state
            if hasattr(state, "spec_info"):
                full_data = make_full_tensor(param, state.spec_info)
                dist.broadcast(full_data, src=loaded)
                copy_to_local(param, full_data, state.spec_info)
            else:
                dist.broadcast(param.data, src=loaded)
        shard_states.pop(param_name)
        del loaded
        return param

    def init_fn(sub_mod: torch.nn.Module):
        """
        Initialize a sub-module with sharded states.

        Args:
            sub_mod (torch.nn.Module): The sub-module to be initialized.

        Returns:
            torch.nn.Module: The initialized sub-module.
        """
        param_and_buffers = tuple(sub_mod.named_parameters(recurse=False)) + tuple(
            sub_mod.named_buffers(recurse=False)
        )
        for name, state in param_and_buffers:
            if state not in state2fqn:
                logger.warning_once(f"{name} in {sub_mod.__class__.__name__} not found in state2fqn, skip it")
                continue
            is_param = name in sub_mod._parameters
            fqn = state2fqn[state].pop(0)
            if (not is_param) and fqn not in shard_states:
                if state.is_meta:
                    raise RuntimeError(
                        f"find a non-persistent buffer ({fqn}) initiated with device meta. "
                        "Such buffer is not saved in checkpoint and user should guarantee to init in CPU / GPU device."
                    )
                continue
            if state in shared:
                if state not in materialized_states:
                    materialized_states[state] = create_and_sync_state(fqn, state, is_param)
                else:
                    if fqn in shard_states:
                        shard_states.pop(fqn)
                materialize_state = materialized_states[state]
            else:
                materialize_state = create_and_sync_state(fqn, state, is_param)
            if is_param:
                sub_mod._parameters[name] = materialize_state
            else:
                sub_mod._buffers[name] = materialize_state
        return sub_mod

    return init_fn


def init_fsdp_fn(model: nn.Module, device: Union[str, "torch.device"]) -> Callable[[nn.Module], None]:
    """
    Gets tensor materialization function that supports shared parameters and buffers.
    Args:
        model (nn.Module): the top module that may include shared parameters / buffers.
        device (Union[str, torch.device]): the device to initialize parameters on.

    Returns:
        Callable[[nn.Module], None]: initialization method to materialize meta tensors on device.
    """
    param_occurrence = defaultdict(int)
    for _, param in model.named_parameters(remove_duplicate=False):
        param_occurrence[param] += 1

    duplicated_params = {param for param in param_occurrence.keys() if param_occurrence[param] > 1}
    materialized_params = {}

    def init_fn(module: "nn.Module"):
        for name, param in module.named_parameters(recurse=False):
            if param in duplicated_params:
                module._parameters[name] = materialized_params.setdefault(
                    param, nn.Parameter(torch.empty_like(param.data, device=device), requires_grad=param.requires_grad)
                )
            else:
                module._parameters[name] = nn.Parameter(
                    torch.empty_like(param.data, device=device), requires_grad=param.requires_grad
                )

    return init_fn
