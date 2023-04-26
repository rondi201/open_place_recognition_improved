"""Basic fusion layers implementation."""
from typing import Dict, Union

import MinkowskiEngine as ME  # noqa: N817
import torch
from torch import Tensor

from opr.models.base_models import FusionModule


class Concat(FusionModule):
    """Concatenation module for 'image' and 'cloud' modalities."""

    def __init__(self) -> None:
        """Concatenation module for 'image' and 'cloud' modalities."""
        super().__init__()

    def forward(self, data: Dict[str, Union[Tensor, ME.SparseTensor]]) -> Tensor:  # noqa: D102
        assert "image" in data
        assert "cloud" in data
        concat_data = []
        for key, value in data.items():
            if "cloud" in key:
                concat_data.append(value)
                continue
            batch, count, features = value.shape
            if count == 1:
                concat_data.append(value.sqeeze(1))
                continue
            concat_data.append(value.reshape(batch, -1))
        fusion_global_descriptor = torch.concat(concat_data, dim=1)
        return fusion_global_descriptor

class LinearConcat(Concat):
    """Concatenation module for 'image' and 'cloud' modalities."""

    def __init__(self, in_features, out_features) -> None:
        """Concatenation module for 'image' and 'cloud' modalities."""
        super().__init__()
        self.head = torch.nn.Sequential(
            torch.nn.Linear(in_features, out_features),
            torch.nn.BatchNorm1d(out_features)
        )

    def forward(self, data: Dict[str, Union[Tensor, ME.SparseTensor]]) -> Tensor:  # noqa: D102
        fusion_global_descriptor = super().forward(data)
        fusion_global_descriptor = self.head(fusion_global_descriptor)

        return fusion_global_descriptor