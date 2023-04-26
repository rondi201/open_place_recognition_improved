"""Interfaces and meta-models definitions."""
from typing import Dict, Optional, Union

import MinkowskiEngine as ME  # noqa: N817
import torch
from torch import Tensor, nn

# ----new ----
from transformers import BertModel

class TextModule(nn.Module):
    def __init__(self, out_channels=256):
        """Interface class for image feature extractor module."""
        super().__init__()
        self.encoder = BertModel.from_pretrained('bert-base-uncased')
        self.head = nn.Sequential(nn.Linear(self.encoder.hidden_size, out_channels),
                                  nn.BatchNorm1d(out_channels),
                                  )
    
    def forward(self, token):
        x = token
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        x = self.model_bert(x)[0][:,0,:]
        x = self.head(x)

        return x


class ImageFeatureExtractor(nn.Module):
    """Interface class for image feature extractor module."""

    def __init__(self):
        """Interface class for image feature extractor module."""
        super().__init__()

    def forward(self, image: Tensor) -> Tensor:  # noqa: D102
        raise NotImplementedError()


class ImageHead(nn.Module):
    """Interface class for image head module."""

    def __init__(self):
        """Interface class for image head module."""
        super().__init__()

    def forward(self, feature_map: Tensor) -> Tensor:  # noqa: D102
        raise NotImplementedError()


class ImageModule(nn.Module):
    """Meta-module for image branch. Combines feature extraction backbone and head modules."""

    def __init__(
        self,
        backbone: ImageFeatureExtractor,
        head: ImageHead,
    ):
        """Meta-module for image branch.

        Args:
            backbone (ImageFeatureExtractor): Image feature extraction backbone.
            head (ImageHead): Image head module.
        """
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        x = self.backbone(x)
        x = self.head(x)
        return x


class CloudFeatureExtractor(nn.Module):
    """Interface class for cloud feature extractor module."""

    sparse: bool

    def __init__(self):
        """Interface class for cloud feature extractor module."""
        super().__init__()
        assert self.sparse is not None

    def forward(self, cloud: Union[Tensor, ME.SparseTensor]) -> Union[Tensor, ME.SparseTensor]:  # noqa: D102
        raise NotImplementedError()


class CloudHead(nn.Module):
    """Interface class for cloud head module."""

    sparse: bool

    def __init__(self):
        """Interface class for cloud head module."""
        super().__init__()
        assert self.sparse is not None

    def forward(self, feature_map: Union[Tensor, ME.SparseTensor]) -> Tensor:  # noqa: D102
        raise NotImplementedError()


class CloudModule(nn.Module):
    """Meta-module for cloud branch. Combines feature extraction backbone and head modules."""

    def __init__(
        self,
        backbone: CloudFeatureExtractor,
        head: CloudHead,
    ):
        """Meta-module for cloud branch.

        Args:
            backbone (CloudFeatureExtractor): Cloud feature extraction backbone.
            head (CloudHead): Cloud head module.

        Raises:
            ValueError: If incompatible cloud backbone and head are given.
        """
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.sparse = self.backbone.sparse
        if self.backbone.sparse != self.head.sparse:
            raise ValueError("Incompatible cloud backbone and head")

    def forward(self, x: Union[Tensor, ME.SparseTensor]) -> Tensor:  # noqa: D102
        if self.sparse:
            assert isinstance(x, ME.SparseTensor)
        else:
            assert isinstance(x, Tensor)
        x = self.backbone(x)
        x = self.head(x)
        return x


class FusionModule(nn.Module):
    """Interface class for fusion module."""

    def __init__(self):
        """Interface class for fusion module."""
        super().__init__()

    def forward(self, data: Dict[str, Union[Tensor, ME.SparseTensor]]) -> Tensor:  # noqa: D102
        raise NotImplementedError()


class ComposedModel(nn.Module):
    """Composition model for multimodal architectures."""

    sparse_cloud: Optional[bool] = None

    def __init__(
        self,
        image_module: Optional[ImageModule] = None,
        cloud_module: Optional[CloudModule] = None,
        fusion_module: Optional[FusionModule] = None,
    ) -> None:
        """Composition model for multimodal architectures.

        Args:
            image_module (ImageModule, optional): Image modality branch. Defaults to None.
            cloud_module (CloudModule, optional): Cloud modality branch. Defaults to None.
            fusion_module (FusionModule, optional): Module to fuse different modalities. Defaults to None.
        """
        super().__init__()

        self.image_module = image_module
        self.cloud_module = cloud_module
        self.fusion_module = fusion_module
        if self.cloud_module:
            self.sparse_cloud = self.cloud_module.sparse

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Optional[Tensor]]:  # noqa: D102
        out_dict: Dict[str, Optional[Tensor]] = {
            "image": None,
            "cloud": None,
            "fusion": None,
        }

        if self.image_module is not None:
            out_dict["image"] = self.image_module(batch["images"])

        if self.cloud_module is not None:
            if self.sparse_cloud:
                cloud = ME.SparseTensor(features=batch["features"], coordinates=batch["coordinates"])
            else:
                raise NotImplementedError("Currently we support only sparse cloud modules.")
            out_dict["cloud"] = self.cloud_module(cloud)

        if self.fusion_module is not None:
            out_dict["fusion"] = self.fusion_module(out_dict)

        return out_dict

class ComposedImprovedModel(nn.Module):
    """Composition model for multimodal architectures."""

    sparse_cloud: Optional[bool] = None

    def __init__(
        self,
        image_module: Optional[ImageModule] = None,
        cloud_module: Optional[CloudModule] = None,
        text_module: Optional[TextModule] = None,
        fusion_module: Optional[FusionModule] = None,
    ) -> None:
        """Composition model for multimodal architectures.

        Args:
            image_module (ImageModule, optional): Image modality branch. Defaults to None.
            cloud_module (CloudModule, optional): Cloud modality branch. Defaults to None.
            fusion_module (FusionModule, optional): Module to fuse different modalities. Defaults to None.
        """
        super().__init__()

        self.image_module = image_module
        self.cloud_module = cloud_module
        self.text_module = text_module
        self.fusion_module = fusion_module
        if self.cloud_module:
            self.sparse_cloud = self.cloud_module.sparse

        self.task2model = {
            "image": self.image_module,
            "cloud": self.cloud_module,
            "text": self.text_module,
        }

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Optional[Tensor]]:  # noqa: D102
        out_dict: Dict[str, Optional[Tensor]] = {
            "image": None,
            "cloud": None,
            "text": None,
            "fusion": None,
        }
        special_task = []

        if self.cloud_module is not None:
            special_task.append("cloud")
            if self.sparse_cloud:
                cloud = ME.SparseTensor(features=batch["features"], coordinates=batch["coordinates"])
            else:
                raise NotImplementedError("Currently we support only sparse cloud modules.")
            out_dict["cloud"] = self.cloud_module(cloud)

        for task in self.task2model.keys():
            if task in special_task or self.task2model[task] is not None:
                continue
            res = self._grouped_task_forward(batch, task)
            if res is not None:
                out_dict[task] = res

        if self.fusion_module is not None:
            out_dict["fusion"] = self.fusion_module(out_dict)

        return out_dict

    def _grouped_task_forward(self, batch: Dict[str, Tensor], task: str):
        task_keys = [key for key in batch.keys() if task in key]
        if len(task_keys) == 0:
            return None
        task_data = [batch[key] for key in task_keys]
        task_vectors = []
        for task_batch in task_data:
            if self.task2model[task] is None:
                return None
            vector = self.task2model[task](task_batch)
            task_vectors.append(vector)
        if len(task_vectors) > 1:
            return torch.stack(task_vectors, dim=1)
        else:
            return task_vectors[0].unsqeeze(1)
