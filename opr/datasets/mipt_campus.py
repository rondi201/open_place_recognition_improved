from pathlib import Path
from typing import Dict, Literal, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch import Tensor

from opr.datasets.augmentations import (
    DefaultCloudSetTransform,
    DefaultCloudTransform,
    DefaultImageTransform,
    DefaultMaskTransform,
)
from opr.datasets.base import BaseDataset


class MIPTCampusDataset(BaseDataset):
    """MIPT Campus dataset implementation."""

    valid_subsets = ("train", "val", "test")
    valid_modalities = ("image", "cloud", "mask", "text")
    images_subdir = {"front": "front_cam", "back": "back_cam"}
    clouds_subdir: Optional[str] = "lidar"
    mask_subdir: Optional[str] = "labels"

    def __init__(
        self,
        dataset_root: Union[str, Path],
        subset: Literal["train", "val", "test"] = "train",
        modalities: Union[str, Tuple[str, ...]] = ("image", "cloud", "mask", "text"),
        cam_types: Optional[Union[str, Tuple[str, ...]]] = ('front', 'back'),
        random_select_nearest_images: bool = False,
        mink_quantization_size: Optional[float] = 0.01,
    ) -> None:
        """MIPT dataset implementation.

        Args:
            dataset_root (Union[str, Path]): Path to the dataset root directory.
            subset (Literal["train", "val", "test"]): Current subset to load. Defaults to "train".
            modalities (Union[str, Tuple[str, ...]]): List of modalities for which the data should be loaded.
                Defaults to ( "front_image", "back_image", "cloud", "front_mask", "back_mask", "front_text", "back_text").
            images_subdir (Union[str, Path], optional): Images subdirectory path. Defaults to "stereo_centre".
            random_select_nearest_images (bool): Whether to select random nearest top-20 images
                as described in "MinkLoc++" paper. Defaults to False.
            mink_quantization_size (float, optional): The quantization size for point clouds.
                Defaults to 0.01.

        Raises:
            ValueError: If images_subdir is undefined when "images" in modalities.
        """
        super().__init__(dataset_root, subset, modalities)

        self.mink_quantization_size = mink_quantization_size

        for cam in cam_types:
            if cam not in self.images_subdir.keys():
                raise ValueError(f'"{cam}" type is not valid')

        if self.subset == "test":
            self.dataset_df["in_query"] = True  # tmp workaround to make it compatible with text_oxford code

        self.cam_types = cam_types

        self.image_transform = DefaultImageTransform(train=(self.subset == "train"))
        self.cloud_transform = DefaultCloudTransform(train=(self.subset == "train"))
        # self.cloud_set_transform = DefaultCloudSetTransform(train=(self.subset == "train"))
        self.mask_transform = DefaultMaskTransform()
        self.text_transform = lambda text: text


    def __getitem__(self, idx: int) -> Dict[str, Union[int, Tensor]]:  # noqa: D105
        # Берём строку из таблицы
        data: Dict[str, Union[int, Tensor]] = {"idx": idx}
        row = self.dataset_df.iloc[idx]

        # Получаем изображения
        data["utm"] = torch.tensor(row[["northing", "easting"]].to_numpy(dtype=np.float32))
        track_dir = self.dataset_root / str(row["track"])
        if "image" in self.modalities and self.images_subdir is not None:
            for cam in self.cam_types:
                im_filepath = track_dir / self.images_subdir[cam] / f"{row[f'{cam}_cam_ts']}.png"
                im = cv2.imread(str(im_filepath))
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                im = self.image_transform(im)
                data[f"{cam}_image"] = im
        if "cloud" in self.modalities and self.clouds_subdir is not None:
            pc_filepath = track_dir / self.clouds_subdir / f"{row['lidar_ts']}.bin"
            pc = self._load_pc(pc_filepath)
            pc = self.cloud_transform(pc)
            data["cloud"] = pc
        import pandas as pd
        if "mask" in self.modalities and self.mask_subdir is not None:
            for cam in self.cam_types:
                im_filepath = track_dir / self.mask_subdir / self.images_subdir[cam] / f"{row[f'{cam}_cam_ts']}.png"
                im = cv2.imread(str(im_filepath))
                # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                im = self.mask_transform(im)
                data[f"{cam}_mask"] = im
        if "text" in self.modalities:
            for cam in self.cam_types:
                text = row[f'{cam}_description']
                text = self.text_transform(text)
                data[f"{cam}_text"] = text
        return data

    def _load_pc(self, filepath: Union[str, Path]) -> Tensor:
        pc = np.fromfile(filepath, dtype=np.float32).reshape((-1, 4))[:, :-1]
        in_range_idx = np.all(
            np.logical_and(-100 <= pc, pc <= 100),  # select points in range [-100, 100] meters
            axis=1,
        )
        pc = pc[in_range_idx]
        pc_tensor = torch.tensor(pc, dtype=torch.float32)
        return pc_tensor
