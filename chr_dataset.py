#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 12:55:53 2024

@author: vishalr
"""
import os
import os.path as osp
import shutil
from typing import Callable, List, Optional

import torch

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.io import read_tu_data


class CHRDataset(InMemoryDataset):
    r"""Description TBA.
    """

    def __init__(self, root: str, name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 use_node_attr: bool = False, use_edge_attr: bool = False,
                 cleaned: bool = False):
        self.name = name
        self.cleaned = cleaned
        super().__init__(root, transform, pre_transform, pre_filter)

        out = torch.load(self.processed_paths[0])
        if not isinstance(out, tuple) or len(out) != 3:
            raise RuntimeError(
                "The 'data' object was created by an older version of PyG. "
                "If this error occurred while loading an already existing "
                "dataset, remove the 'processed/' directory in the dataset's "
                "root folder and try again.")
        data, self.slices, self.sizes = out
        self.data = Data.from_dict(data) if isinstance(data, dict) else data

        if self._data.x is not None and not use_node_attr:
            num_node_attributes = self.num_node_attributes
            self._data.x = self._data.x[:, num_node_attributes:]
        if self._data.edge_attr is not None and not use_edge_attr:
            num_edge_attrs = self.num_edge_attributes
            self._data.edge_attr = self._data.edge_attr[:, num_edge_attrs:]

    @property
    def raw_dir(self) -> str:
        name = f'raw'
        return osp.join(self.root, self.name, name)

    @property
    def processed_dir(self) -> str:
        name = f'processed'
        return osp.join(self.root, self.name, name)

    @property
    def num_node_labels(self) -> int:
        return self.sizes['num_node_labels']

    @property
    def num_node_attributes(self) -> int:
        return self.sizes['num_node_attributes']

    @property
    def num_edge_labels(self) -> int:
        return self.sizes['num_edge_labels']

    @property
    def num_edge_attributes(self) -> int:
        return self.sizes['num_edge_attributes']

    @property
    def raw_file_names(self) -> List[str]:
        names = ['A', 'graph_indicator']
        return [f'{self.name}_{name}.txt' for name in names]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        # for now we assume that the raw folder already exists
        print('The files are already there.')

    def process(self):
        self.data, self.slices, sizes = read_tu_data(self.raw_dir, self.name)

        if self.pre_filter is not None or self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            self.data, self.slices = self.collate(data_list)
            self._data_list = None  # Reset cache.

        torch.save((self._data.to_dict(), self.slices, sizes),
                   self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'