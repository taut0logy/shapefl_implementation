"""
Data module for ShapeFL implementation.
"""

from .data_loader import (
    load_fmnist_data,
    create_non_iid_partitions,
    get_node_dataloader,
    save_partitions,
    load_partitions,
)

__all__ = [
    "load_fmnist_data",
    "create_non_iid_partitions",
    "get_node_dataloader",
    "save_partitions",
    "load_partitions",
]
