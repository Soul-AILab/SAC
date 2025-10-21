import os
from pathlib import Path
from typing import List, Dict
from functools import partial
from omegaconf import DictConfig

import utils.log as log
from utils import data_processor
from models.base.base_datapipes import RawDataset


class BaseDataset(object):
    """Initialize the dataset with the given configuration.

    Args:
        config (DictConfig):
            Configuration dictionary specifying dataset parameters.
        mode (str):
            Dataset mode, typically 'train' or 'val'.
    """

    def __init__(
        self,
        config: DictConfig,
        mode: str = "train",
        prefetch: int = 1000,
        **kwargs,
    ) -> None:
        self.config = config
        self.mode = mode
        self.train = mode == "train"
        if mode != "train":
            self.partition = False
        else:
            self.partition = self.config["dataloader"]["partition"]
        self.prefetch = prefetch

    def load_data(self) -> RawDataset:
        """Load data using an IterDataPipe with configuration settings.

        Returns:
            RawDataset: An instance of RawDataset configured per the specified settings.
        """
        cycle = self.config["dataloader"].get("cycle", 1) if self.mode == "train" else 1
        list_shuffle = self.config["dataloader"].get("list_shuffle", True)
        list_shuffle_size = self.config["dataloader"].get("list_shuffle_size", 10000)

        if self.mode == "train":
            jsonlfiles = self.config["datasets"]["train"]
        else:
            jsonlfiles = self.config["datasets"]["val"]
            list_shuffle = False

        return RawDataset(
            file_paths=jsonlfiles,
            prefetch=self.prefetch,
            partition=self.partition,
            shuffle=list_shuffle,
            shuffle_size=list_shuffle_size,
            cycle=cycle,
        )

    def fetch_data(self, elem: Dict) -> Dict:
        """Fetch a single sample.

        Args:
            elem (Dict): A dictionary with the key 'index' as index.

        Returns:
            Dict:
        """
        index = elem["index"]
        try:
            return {"index": index}
        except Exception as e:
            log.warn(f"Error: {e}")
            return {"index": index}

    def filter(self, elem: dict):
        """Filter out bad data. Return True if the data is kept."""

        return True

    def sample(self):
        """Process and filter the dataset based on defined criteria.

        Returns:
            RawDataset: A processed and potentially filtered dataset ready for training or validation.
        """
        dataset = self.load_data()
        dataset = dataset.map(self.fetch_data)
        dataset = dataset.filter(self.filter)
        # local shuffle
        if self.config["dataloader"].get("shuffle", False):
            dataset = dataset.shuffle(
                buffer_size=self.config["dataloader"]["shuffle_size"]
            )

        # sort based on the length of {sort_key}
        if self.config["dataloader"].get("sort", False):
            dataset = dataset.sort(
                buffer_size=self.config["dataloader"]["sort_size"],
                key_func=partial(
                    data_processor.feats_length_fn,
                    length_key=self.config["dataloader"]["sort_key"],
                ),
            )

        # batch type setting
        batch_type = self.config["dataloader"].get("batch_type", "static")
        assert batch_type in ["static", "bucket", "dynamic"]
        if batch_type == "static":
            batch_size = (
                self.config["dataloader"]["static"]["batch_size"] if self.train else 1
            )
            dataset = dataset.batch(batch_size, wrapper_class=self.padding)

        elif batch_type == "bucket":
            length_key = self.config["dataloader"]["bucket"]["length_key"]
            dataset = dataset.bucket_by_sequence_length(
                partial(data_processor.feats_length_fn, length_key=length_key),
                self.config["dataloader"]["bucket"]["bucket_boundaries"],
                (
                    self.config["dataloader"]["bucket"]["bucket_batch_sizes"]
                    if self.train
                    else 1
                ),
                wrapper_class=self.padding,
            )
        else:
            length_key = self.config["dataloader"]["dynamic"]["length_key"]
            max_length_in_batch = self.config["dataloader"]["dynamic"][
                "max_length_in_batch"
            ]

            dataset = dataset.dynamic_batch(
                data_processor.DynamicBatchWindow(max_length_in_batch, length_key),
                wrapper_class=self.padding,
            )

        return dataset

    def padding(self, batch: List[dict]):
        """Padding the batch data into training data

        Args:
            batch (List[dict])
        """

        assert isinstance(batch, list)
        collate_batch = {}

        for k in batch[0].keys():
            collate_batch[k] = [b[k] for b in batch]

        return collate_batch


# test
if __name__ == "__main__":
    data_root  = '/path/to/test_data'  # please change to your data root
    config = {
        "dataloader": {
            "static": {"batch_size": 10},
            "cycle": 1,
            "list_shuffle": True,
            "list_shuffle_size": 10000,
            "partition": True,
            "shuffle": False,
        },
        "datasets": {
            "train": [f"{data_root}/test_data_part_aa",
                      f"{data_root}/test_data_part_ab",
                      f"{data_root}/test_data_part_ac"],
            "val": [f"{data_root}/test_data_part_ad",
                      f"{data_root}/test_data_part_ae"],
        },
    }

    dataset_sampler = BaseDataset(config)
    dataset = dataset_sampler.sample()

    i = 0
    for batch in dataset:
        i += 1
        print(f"itr {i}, batch_size:", len(batch['index']))

    print("+--------------------+\n|   Test Successful  |\n+--------------------+")
