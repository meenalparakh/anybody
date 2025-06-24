# This file is part of AnyBody (BSD-3-Clause License).
#
# Copyright (c) Meenal Parakh, 2025
# All rights reserved.
#
# This file includes modified code from:
# SKRL (https://github.com/Toni-SM/skrl)
# Licensed under the MIT License (see LICENSES/MIT_LICENSE.txt)
#
# SPDX-License-Identifier: BSD-3-Clause

import os
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import datetime
import csv

# from skrl.memories.torch import Memory
from torch.utils.data.sampler import BatchSampler

from tensordict import TensorDict
from .base import Memory

class CustomMemory(Memory):
    def __init__(
        self,
        memory_size: int,
        num_envs: int = 1,
        device: Optional[Union[str, torch.device]] = None,
        export: bool = False,
        export_format: str = "pt",
        export_directory: str = "",
        replacement=True,
    ) -> None:
        """Random sampling memory

        Sample a batch from memory randomly

        :param memory_size: Maximum number of elements in the first dimension of each internal storage
        :type memory_size: int
        :param num_envs: Number of parallel environments (default: ``1``)
        :type num_envs: int, optional
        :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda"`` if available or ``"cpu"``
        :type device: str or torch.device, optional
        :param export: Export the memory to a file (default: ``False``).
                       If True, the memory will be exported when the memory is filled
        :type export: bool, optional
        :param export_format: Export format (default: ``"pt"``).
                              Supported formats: torch (pt), numpy (np), comma separated values (csv)
        :type export_format: str, optional
        :param export_directory: Directory where the memory will be exported (default: ``""``).
                                 If empty, the agent's experiment directory will be used
        :type export_directory: str, optional
        :param replacement: Flag to indicate whether the sample is with or without replacement (default: ``True``).
                            Replacement implies that a value can be selected multiple times (the batch size is always guaranteed).
                            Sampling without replacement will return a batch of maximum memory size if the memory size is less than the requested batch size
        :type replacement: bool, optional

        :raises ValueError: The export format is not supported
        """
        super().__init__(
            memory_size, num_envs, device, export, export_format, export_directory
        )

        self._replacement = replacement

    def create_tensor(
        self,
        name: str,
        size,
        dtype: Optional[torch.dtype] = None,
        keep_dimensions: bool = False,
    ) -> bool:
        """Create a new internal tensor in memory

        The tensor will have a 3-components shape (memory size, number of environments, size).
        The internal representation will use _tensor_<name> as the name of the class property

        :param name: Tensor name (the name has to follow the python PEP 8 style)
        :type name: str
        :param size: Number of elements in the last dimension (effective data size).
                     The product of the elements will be computed for sequences or gym/gymnasium spaces
        :type size: int, tuple or list of integers, gym.Space, or gymnasium.Space
        :param dtype: Data type (torch.dtype) (default: ``None``).
                      If None, the global default torch data type will be used
        :type dtype: torch.dtype or None, optional
        :param keep_dimensions: Whether or not to keep the dimensions defined through the size parameter (default: ``False``)
        :type keep_dimensions: bool, optional

        :raises ValueError: The tensor name exists already but the size or dtype are different

        :return: True if the tensor was created, otherwise False
        :rtype: bool
        """

        # create tensor for dict spaces, in particular directly checking for the states key
        # if name == "states":  # and isinstance(size, dict):

        if name in ["states", "next_states"]:
            for k, v in size.items():
                self.create_tensor(
                    name=f"{name}_{k}",
                    size=v,
                    dtype=dtype,
                    keep_dimensions=True,
                )
            return True

        # compute data size
        try:    
            size = self._get_space_size(size, keep_dimensions)
        except ValueError:
            import pdb; pdb.set_trace()
        
        # check dtype and size if the tensor exists
        if name in self.tensors:
            tensor = self.tensors[name]
            if keep_dimensions:
                if tensor.shape[2:] != size:
                    raise ValueError(
                        f"Size of tensor {name} ({size}) doesn't match the existing one ({tensor.shape[2:]})"
                    )
            else:
                if tensor.size(-1) != size:
                    raise ValueError(
                        f"Size of tensor {name} ({size}) doesn't match the existing one ({tensor.size(-1)})"
                    )
            if dtype is not None and tensor.dtype != dtype:
                raise ValueError(
                    f"Dtype of tensor {name} ({dtype}) doesn't match the existing one ({tensor.dtype})"
                )
            return False
        # define tensor shape
        tensor_shape = (
            (self.memory_size, self.num_envs, *size)
            if keep_dimensions
            else (self.memory_size, self.num_envs, size)
        )
        view_shape = (-1, *size) if keep_dimensions else (-1, size)
        # create tensor (_tensor_<name>) and add it to the internal storage
        setattr(
            self,
            f"_tensor_{name}",
            torch.zeros(tensor_shape, device=self.device, dtype=dtype),
        )
        # update internal variables
        self.tensors[name] = getattr(self, f"_tensor_{name}")
        self.tensors_view[name] = self.tensors[name].view(*view_shape)
        self.tensors_keep_dimensions[name] = keep_dimensions
        # fill the tensors (float tensors) with NaN
        for tensor in self.tensors.values():
            if torch.is_floating_point(tensor):
                tensor.fill_(float("nan"))
        return True

    def add_samples(self, **tensors: torch.Tensor) -> None:
        """Record samples in memory

        Samples should be a tensor with 2-components shape (number of environments, data size).
        All tensors must be of the same shape

        According to the number of environments, the following classification is made:

        - one environment:
          Store a single sample (tensors with one dimension) and increment the environment index (second index) by one

        - number of environments less than num_envs:
          Store the samples and increment the environment index (second index) by the number of the environments

        - number of environments equals num_envs:
          Store the samples and increment the memory index (first index) by one

        :param tensors: Sampled data as key-value arguments where the keys are the names of the tensors to be modified.
                        Non-existing tensors will be skipped
        :type tensors: dict

        :raises ValueError: No tensors were provided or the tensors have incompatible shapes
        """
        if not tensors:
            raise ValueError(
                "No samples to be recorded in memory. Pass samples as key-value arguments (where key is the tensor name)"
            )

        # flattening the states which is a dict itself
        if "states" in tensors and (isinstance(tensors["states"], (dict, TensorDict))):
            state_tensors_dict = tensors["states"]
            state_key_names = list(state_tensors_dict.keys())
            for k in state_key_names:
                tensors[f"states_{k}"] = state_tensors_dict[k]

        if "next_states" in tensors and (isinstance(tensors["next_states"], (dict, TensorDict))):
            next_state_tensors_dict = tensors["next_states"]
            next_state_key_names = list(next_state_tensors_dict.keys())
            for k in next_state_key_names:
                if f'next_states_{k}' in self.tensors:
                    tensors[f"next_states_{k}"] = next_state_tensors_dict[k]
            

            # tensors.pop("states")
            # no need to pop states, as the below checks if the name is in self.tensor or not,
            # for "states" no such key was created in self.tensors, as we created "states_{k}" keys

        # dimensions and shapes of the tensors (assume all tensors have the dimensions of the first tensor)
        tmp = tensors.get(
            "actions", tensors[next(iter(tensors))]
        )  # ask for states first
        dim, shape = tmp.ndim, tmp.shape

        # multi environment (number of environments equals num_envs)
        
        # # print the dimensions of the buffer and the samples to add
        # for name, tensor in tensors.items():
        #     if name in self.tensors:
        #         print(f"{name}: buffer shape: {self.tensors[name].shape}, sample shape: {tensor.shape}")
        
        
        if dim == 2 and shape[0] == self.num_envs:
            for name, tensor in tensors.items():
                if name in self.tensors:
                    try:           
                        self.tensors[name][self.memory_index].copy_(tensor)
                    except Exception as e:
                        print(f"Error: {e}")
                    # except TypeError:
                        import pdb; pdb.set_trace()
                        

            self.memory_index += 1
        # multi environment (number of environments less than num_envs)
        elif dim == 2 and shape[0] < self.num_envs:
            for name, tensor in tensors.items():
                if name in self.tensors:
                    self.tensors[name][
                        self.memory_index,
                        self.env_index : self.env_index + tensor.shape[0],
                    ].copy_(tensor)
            self.env_index += tensor.shape[0]
        # single environment - multi sample (number of environments greater than num_envs (num_envs = 1))
        elif dim == 2 and self.num_envs == 1:
            for name, tensor in tensors.items():
                if name in self.tensors:
                    num_samples = min(shape[0], self.memory_size - self.memory_index)
                    remaining_samples = shape[0] - num_samples
                    # copy the first n samples
                    self.tensors[name][
                        self.memory_index : self.memory_index + num_samples
                    ].copy_(tensor[:num_samples].unsqueeze(dim=1))
                    self.memory_index += num_samples
                    # storage remaining samples
                    if remaining_samples > 0:
                        self.tensors[name][:remaining_samples].copy_(
                            tensor[num_samples:].unsqueeze(dim=1)
                        )
                        self.memory_index = remaining_samples
        # single environment
        elif dim == 1:
            for name, tensor in tensors.items():
                if name in self.tensors:
                    self.tensors[name][self.memory_index, self.env_index].copy_(tensor)
            self.env_index += 1
        else:
            raise ValueError(
                f"Expected shape (number of environments = {self.num_envs}, data size), got {shape}"
            )

        # update indexes and flags
        if self.env_index >= self.num_envs:
            self.env_index = 0
            self.memory_index += 1
        if self.memory_index >= self.memory_size:
            self.memory_index = 0
            self.filled = True

            # export tensors to file
            # if self.export:
            #     print("Exporting memory to file", self.export_directory)
            #     self.save(directory=self.export_directory, format=self.export_format)

    def save_till_index(self, index: int = 0, directory: str = "", format: str = "pt") -> None:
        """Save the memory to a file

        Supported formats:

        - PyTorch (pt)
        - NumPy (npz)
        - Comma-separated values (csv)

        :param directory: Path to the folder where the memory will be saved.
                          If not provided, the directory defined in the constructor will be used
        :type directory: str
        :param format: Format of the file where the memory will be saved (default: ``"pt"``)
        :type format: str, optional

        :raises ValueError: If the format is not supported
        """
        if not directory:
            directory = self.export_directory
        os.makedirs(os.path.join(directory, "memories"), exist_ok=True)
        memory_path = os.path.join(directory, "memories", "{}_memory_{}.{}".format(datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S-%f"), hex(id(self)), format))

        if index == 0:
            index = self.memory_size

        # torch
        if format == "pt":
            torch.save({name: self.tensors[name][:index].detach().cpu() for name in self.get_tensor_names()}, memory_path)
        elif format == "npz":
            
            # save the tensors in pairs of two environments
            for env_id in range(0, self.num_envs, 2):
                memory_path = os.path.join(directory, "memories", "{}_memory_{}_{}.{}".format(datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S-%f"), env_id, hex(id(self)), format))
                np.savez(
                    memory_path,
                    **{
                        name: self.tensors[name][:index, env_id : env_id + 2].cpu().numpy()
                        for name in self.get_tensor_names()
                    }
                )
            
            np.savez(memory_path, **{name: self.tensors[name][:index].cpu().numpy() for name in self.get_tensor_names()})
        else:
            raise NotImplementedError(f"Unsupported format: {format}. Available formats: pt")

    def sample(
        self,
        names: Tuple[str],
        batch_size: int,
        mini_batches: int = 1,
        sequence_length: int = 1,
    ) -> List[List[torch.Tensor]]:
        """Sample a batch from memory randomly

        :param names: Tensors names from which to obtain the samples
        :type names: tuple or list of strings
        :param batch_size: Number of element to sample
        :type batch_size: int
        :param mini_batches: Number of mini-batches to sample (default: ``1``)
        :type mini_batches: int, optional
        :param sequence_length: Length of each sequence (default: ``1``)
        :type sequence_length: int, optional

        :return: Sampled data from tensors sorted according to their position in the list of names.
                 The sampled tensors will have the following shape: (batch size, data size)
        :rtype: list of torch.Tensor list
        """
        # compute valid memory sizes
        size = len(self)
                
        if sequence_length > 1:
            sequence_indexes = torch.arange(
                0, self.num_envs * sequence_length, self.num_envs
            )
            size -= sequence_indexes[-1].item()

        # generate random indexes
        if self._replacement:
            indexes = torch.randint(0, size, (batch_size,))
        else:
            # details about the random sampling performance can be found here:
            # https://discuss.pytorch.org/t/torch-equivalent-of-numpy-random-choice/16146/19
            indexes = torch.randperm(size, dtype=torch.long)[:batch_size]

        # generate sequence indexes
        if sequence_length > 1:
            indexes = (
                sequence_indexes.repeat(indexes.shape[0], 1) + indexes.view(-1, 1)
            ).view(-1)

        self.sampling_indexes = indexes
        return self.sample_by_index(
            names=names, indexes=indexes, mini_batches=mini_batches
        )

    def sample_state_key(self, indices):
        state_keys = [k[7:] for k in self.tensors.keys() if k.startswith("states_")]
        return TensorDict(
            {k: self.tensors_view[f"states_{k}"][indices] for k in state_keys}
        )

    def my_sample_by_index(self, names, indices=None):
        result = []

        if indices is not None:
            for name in names:
                if name in ['states', 'next_states']:
                    state_keys = [
                        k[len(name) + 1:] for k in self.tensors.keys() if k.startswith(f"{name}_")
                    ]
                    result.append(
                        TensorDict(
                            {
                                k: self.tensors_view[f"{name}_{k}"][indices]
                                for k in state_keys
                            }
                        )
                    )
                else:
                    result.append(self.tensors_view[name][indices])

        else:
            for name in names:
                if name in ['states', 'next_states']:
                    state_keys = [
                        k[len(name) + 1:] for k in self.tensors.keys() if k.startswith(f"{name}_")
                    ]
                    result.append(
                        TensorDict(
                            {k: self.tensors_view[f"{name}_{k}"] for k in state_keys}
                        )
                    )
                else:
                    result.append(self.tensors_view[name])

        return result

    def sample_by_index(
        self,
        names: Tuple[str],
        indexes: Union[tuple, np.ndarray, torch.Tensor],
        mini_batches: int = 1,
    ) -> List[List[torch.Tensor]]:
        """Sample data from memory according to their indexes

        :param names: Tensors names from which to obtain the samples
        :type names: tuple or list of strings
        :param indexes: Indexes used for sampling
        :type indexes: tuple or list, numpy.ndarray or torch.Tensor
        :param mini_batches: Number of mini-batches to sample (default: ``1``)
        :type mini_batches: int, optional

        :return: Sampled data from tensors sorted according to their position in the list of names.
                 The sampled tensors will have the following shape: (number of indexes, data size)
        :rtype: list of torch.Tensor list
        """

        if mini_batches > 1:
            batches = BatchSampler(
                indexes, batch_size=len(indexes) // mini_batches, drop_last=True
            )
            result = [self.my_sample_by_index(names, batch) for batch in batches]
            return result

        else:
            return [self.my_sample_by_index(names, indexes)]


    def sample_all(
        self, names: Tuple[str], mini_batches: int = 1, sequence_length: int = 1
    ) -> List[List[torch.Tensor]]:
        """Sample all data from memory

        :param names: Tensors names from which to obtain the samples
        :type names: tuple or list of strings
        :param mini_batches: Number of mini-batches to sample (default: ``1``)
        :type mini_batches: int, optional
        :param sequence_length: Length of each sequence (default: ``1``)
        :type sequence_length: int, optional

        :return: Sampled data from memory.
                 The sampled tensors will have the following shape: (memory size * number of environments, data size)
        :rtype: list of torch.Tensor list
        """
        # sequential order
        if sequence_length > 1:
            if mini_batches > 1:
                batches = BatchSampler(
                    self.all_sequence_indexes,
                    batch_size=len(self.all_sequence_indexes) // mini_batches,
                    drop_last=True,
                )
                return [self.my_sample_by_index(names, batch) for batch in batches]
            return [self.my_sample_by_index(names, self.all_sequence_indexes)]

        # default order
        if mini_batches > 1:
            indexes = np.arange(self.memory_size * self.num_envs)
            batches = BatchSampler(
                indexes, batch_size=len(indexes) // mini_batches, drop_last=True
            )
            return [self.my_sample_by_index(names, batch) for batch in batches]

        return [self.my_sample_by_index(names, None)]
