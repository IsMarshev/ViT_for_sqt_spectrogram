import os
from typing import Dict, Literal, Tuple
from time import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from scipy.interpolate import interp1d

from pretrain.data_model import BatchDict
from utils import bcolors

class CoverDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        file_ext: str,
        dataset_path: str,
        data_split: Literal["train", "val", "test"],
        debug: bool,
        max_len: int,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.file_ext = file_ext
        self.dataset_path = dataset_path
        self.data_split = data_split
        self.debug = debug
        self.max_len = max_len
        self._load_data()
        self.current_index = 0
        self.mask_fraction = 0.15
        self.patch_size = (10,10)
        self.stretch_factor = 1.68


    def __len__(self) -> int:
        return len(self.track_ids)

    def __getitem__(self, index: int) -> BatchDict:
        track_id = self.track_ids[index]
        spectrogram, mask = self._load_cqt(track_id)
    
        return dict(
            spectrogram=spectrogram,
            mask = mask
        )

    def _make_file_path(self, track_id, file_ext):
        a = track_id % 10
        b = track_id // 10 % 10
        c = track_id // 100 % 10
        return os.path.join(str(c), str(b), str(a), f'{track_id}.{file_ext}')

    def _load_data(self) -> None:
        if self.data_split in ['train', 'val']:
            cliques_subset = np.load(os.path.join(self.data_path, "splits", "{}_cliques.npy".format(self.data_split)).replace('\\','/'))
            self.versions = pd.read_csv(
                os.path.join(self.data_path, "cliques2versions.tsv"), sep='\t', converters={"versions": eval}
            )
            self.versions = self.versions[self.versions["clique"].isin(set(cliques_subset))]
            mapping = {}
            for k, clique in enumerate(sorted(cliques_subset)):
                mapping[clique] = k
            self.versions["clique"] = self.versions["clique"].map(lambda x: mapping[x])
            self.versions.set_index("clique", inplace=True)
            self.version2clique = pd.DataFrame(
                [{'version': version, 'clique': clique} for clique, row in self.versions.iterrows() for version in row['versions']]
            ).set_index('version')
            self.track_ids = self.version2clique.index.to_list()
        else:
            self.track_ids = np.load(os.path.join(self.data_path, "splits", "{}_ids.npy".format(self.data_split)))

    def mask_random_patches(self, spectrogram):
        masked_spectrogram = spectrogram.copy()
        num_patches = int(self.mask_fraction * (spectrogram.shape[0] * spectrogram.shape[1]) // (self.patch_size[0] * self.patch_size[1]))
        for _ in range(num_patches):
            i = np.random.randint(0, spectrogram.shape[0] - self.patch_size[0])
            j = np.random.randint(0, spectrogram.shape[1] - self.patch_size[1])
            masked_spectrogram[i:i + self.patch_size[0], j:j + self.patch_size[1]] = 0 
        return masked_spectrogram
    
    def mask_image(self, spectrogram, mask_ratio=0.65):
        H, W = spectrogram.shape
        num_masked_pixels = int(mask_ratio * H * W)

        mask_indices = np.random.permutation(H * W)[:num_masked_pixels]
        mask = np.ones((H, W), dtype=spectrogram.dtype)

        mask.flat[mask_indices] = 0
        masked_spectrogram = spectrogram * mask

        return masked_spectrogram
    
    
    
    def stretch_signal(self, spectrogram):
        num_rows, num_cols = spectrogram.shape
        original_time = np.linspace(0, num_cols - 1, num_cols)
        new_length = int(num_cols * self.stretch_factor)
        new_time = np.linspace(0, num_cols - 1, new_length)

        stretched_spectrogram = np.zeros((num_rows, new_length))

        for i in range(num_rows):
            interpolator = interp1d(original_time, spectrogram[i, :], kind='cubic', fill_value='extrapolate')
            stretched_spectrogram[i, :] = interpolator(new_time)
        
        return stretched_spectrogram


    def add_white_noise(self, signal):
        noise_level = np.random.uniform(0.05, 0.15)
        signal_mean = abs(signal.mean(1).mean(0))
        noise = np.random.normal(0, noise_level*signal_mean, signal.shape)
        return signal + noise

    def _load_cqt(self, track_id: str) -> torch.Tensor:
        filename = os.path.join(self.dataset_path, self._make_file_path(track_id, self.file_ext))
        cqt_spectrogram = np.load(filename)

        cqt_spectrogram = self.stretch_signal(cqt_spectrogram)

        cqt_spectrogram = self.add_white_noise(cqt_spectrogram)

        masked_spectrogram = self.mask_image(cqt_spectrogram)     

        cqt_spectrogram = torch.tensor(cqt_spectrogram, dtype=torch.float32)
        masked_spectrogram = torch.tensor(masked_spectrogram, dtype=torch.float32) 

        return cqt_spectrogram, masked_spectrogram
    
    
    
    



def cover_dataloader(
    data_path: str,
    file_ext: str,
    dataset_path: str,
    data_split: Literal["train", "val", "test"],
    debug: bool,
    max_len: int,
    batch_size: int,
    **config: Dict,
) -> DataLoader:
    return DataLoader(
        CoverDataset(data_path, file_ext, dataset_path, data_split, debug, max_len=max_len),
        batch_size=batch_size if max_len > 0 else 1,
        num_workers=config["num_workers"],
        shuffle=config["shuffle"],
        drop_last=config["drop_last"],
    )