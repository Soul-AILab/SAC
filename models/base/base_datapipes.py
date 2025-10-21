import sys
import json
import copy
import collections
import torch
from typing import Union, List
from pathlib import Path
from collections.abc import Callable

from torch.utils.data import datapipes, IterDataPipe, functional_datapipe
from torch.utils.data.datapipes.iter import Mapper, Grouper
from torch.utils.data.datapipes.iter.sharding import (
    SHARDING_PRIORITIES,
    ShardingFilterIterDataPipe,
)
from torch.utils.data.datapipes.utils.common import _check_unpickable_fn
import utils.log as log


@functional_datapipe("map_ignore_error")
class MapperIgnoreError(Mapper):
    """
    Extends the Mapper class to ignore errors during mapping, optionally logging them.

    Args:
        source_pipe (IterDataPipe):
            The source IterDataPipe from which elements are drawn.
        transformation (Callable):
            A callable applied to each element; errors in this function are handled gracefully.
        input_col, output_col:
            Specify the input and output columns if the elements are dictionaries.
        log_errors (bool):
            If True, logs errors encountered during the transformation.
    """

    def __init__(
        self,
        source_pipe: IterDataPipe,
        transformation: Callable,
        input_col=None,
        output_col=None,
        log_errors: bool = True,
    ) -> None:
        super().__init__(source_pipe, transformation, input_col, output_col)
        self._iter = None
        self.log_errors = log_errors

    def __iter__(self):
        if self._iter is None:
            self._iter = iter(self.datapipe)

        while True:
            try:
                elem = next(self._iter)
                yield self._apply_fn(elem)
            except StopIteration:
                self._iter = None
                return
            except Exception as ex:
                if self.log_error:
                    log.warning(str(ex))


@functional_datapipe("bucket_by_sequence_length")
class BucketBySequenceLength(IterDataPipe):
    """
    Groups elements into buckets based on sequence length constraints.

    Args:
        source_pipe (IterDataPipe):
            The IterDataPipe where elements come from.
        length_func (Callable):
            Function to calculate the length of each element.
        boundaries (List[int]):
            List of boundaries defining bucket ranges.
        batch_sizes (List[int]):
            How many items each batch in the bucket should contain.
        wrapper_class:
            Optional; wraps the batched elements before yielding.
    """

    def __init__(
        self,
        source_pipe: IterDataPipe,
        length_func: Callable,
        boundaries: List[int],
        batch_sizes: List[int],
        wrapper_class=None,
    ) -> None:
        super().__init__()
        _check_unpickable_fn(length_func)
        assert len(batch_sizes) == len(boundaries) + 1
        self.length_func = length_func
        self.boundaries = boundaries + [sys.maxsize]
        self.batch_sizes = batch_sizes
        self.group_pipe = GroupByWindow(
            source_pipe,
            key_fn=self._get_bucket_id,
            window_size_fn=self._get_window_size,
            wrapper_class=wrapper_class,
        )

    def __iter__(self):
        yield from self.group_pipe

    def _get_bucket_id(self, element):
        length = self.length_func(element)
        bucket_id = 0
        for i, boundary in enumerate(self.boundaries):
            if length < boundary:
                bucket_id = i
                break
        return bucket_id

    def _get_window_size(self, bucket_id):
        return self.batch_sizes[bucket_id]


@functional_datapipe("group_by_window")
class GroupByWindow(Grouper):
    """
    Groups elements based on a window size function.

    Args:
        source_pipe:
            The source IterDataPipe.
        key_fn:
            Function to determine the grouping key for each element.
        window_size_fn:
            Function to determine the size of each group.
        wrapper_class:
            Optional; function to process elements before yielding the group.
    """

    def __init__(
        self, source_pipe: IterDataPipe, key_fn, window_size_fn, wrapper_class=None
    ):
        super().__init__(
            source_pipe, key_fn, keep_key=False, group_size=None, drop_remaining=False
        )
        _check_unpickable_fn(window_size_fn)
        self.dp = source_pipe
        self.window_size_fn = window_size_fn
        if wrapper_class is not None:
            _check_unpickable_fn(wrapper_class)
            del self.wrapper_class
            self.wrapper_class = wrapper_class

    def __iter__(self):
        for element in self.datapipe:
            key = self.group_key_fn(element)
            self.buffer_elements[key].append(element)

            self.curr_buffer_size += 1

            group_size = self.window_size_fn(key)
            if group_size == len(self.buffer_elements[key]):
                result = self.wrapper_class(self.buffer_elements[key])
                yield result
                self.curr_buffer_size -= len(self.buffer_elements[key])
                del self.buffer_elements[key]

            if self.curr_buffer_size == self.max_buffer_size:
                result_to_yield = self._remove_biggest_key()
                if result_to_yield is not None:
                    result = self.wrapper_class(result_to_yield)
                    yield result

        for key in tuple(self.buffer_elements.keys()):
            result = self.wrapper_class(self.buffer_elements.pop(key))
            self.curr_buffer_size -= len(result)
            yield result


@functional_datapipe("sort")
class Sort(IterDataPipe):
    """
    Sorts elements in a data pipe.

    Args:
        source_pipe:
            Source IterDataPipe providing the elements to be sorted.
        buffer_size:
            Maximum number of elements held before sorting and yielding them.
        key_fn:
            Function to sort the elements; can depend on element properties.
        reverse (bool):
            If True, sorts in descending order.
    """

    def __init__(
        self,
        source_pipe: IterDataPipe,
        buffer_size: int = 500,
        key_fn=None,
        reverse=False,
    ):
        if key_fn is not None:
            _check_unpickable_fn(key_fn)
        self.buffer_size = buffer_size
        super().__init__()
        self.dp = source_pipe
        self._buffer = []
        self.key_func = key_fn
        self.reverse = reverse

    def __iter__(self):
        for elem in self.dp:
            self._buffer.append(elem)
            if len(self._buffer) >= self.buffer_size:
                self._buffer.sort(key=self.key_func, reverse=self.reverse)
                for x in self._buffer:
                    yield x
                del self._buffer
                self._buffer = []
        # The sample left over
        self._buffer.sort(key=self.key_func, reverse=self.reverse)
        for x in self._buffer:
            yield x
        del self._buffer
        self._buffer = []


@functional_datapipe("dynamic_batch")
class DynamicBatch(IterDataPipe):
    """
    Dynamically batches elements based on a condition.

    Args:
        source_pipe:
            IterDataPipe providing elements.
        window_class:
            Function that determines whether to batch the current element with the buffer.
        wrapper_class:
            Function to wrap the batch before yielding.
    """

    def __init__(
        self, source_pipe: IterDataPipe, window_class: Callable, wrapper_class: Callable
    ) -> None:
        _check_unpickable_fn(window_class)
        _check_unpickable_fn(wrapper_class)
        super().__init__()
        self.dp = dataset
        assert window_class is not None
        assert wrapper_class is not None
        self.window_class = window_class
        self._buffer = []
        self._wrappr_class = wrapper_class

    def __iter__(self):
        for elem in self.dp:
            if not self.window_class(elem, len(self._buffer)):
                self._buffer.append(elem)
            else:
                if len(self._buffer) > 0:
                    yield self._wrappr_class(self._buffer)
                del self._buffer
                self._buffer = [elem]
        if len(self._buffer) > 0:
            yield self._wrappr_class(self._buffer)
        del self._buffer
        self._buffer = []


@functional_datapipe("prefetch")
class Prefetch(IterDataPipe):
    """
    Prefetches elements to improve data loading performance.

    Args:
        source_pipe:
            Source IterDataPipe from which elements are fetched.
        buffer_size:
            Size of the buffer used to prefetch elements.
    """

    def __init__(self, source_pipe: IterDataPipe, buffer_size: int = 500):
        super().__init__()
        self.dp = source_pipe
        self._iter = None
        self._prefetch_buffer_size = buffer_size
        self._buffer = None
        if self._prefetch_buffer_size > 0:
            self._buffer = collections.deque(maxlen=self._prefetch_buffer_size)

    def __iter__(self):
        if self._prefetch_buffer_size > 0:
            if self._iter is None:
                self._iter = iter(self.dp)
            assert self._buffer is not None

            while True:
                if len(self._buffer) <= self._prefetch_buffer_size // 2:
                    while len(self._buffer) < self._prefetch_buffer_size:
                        try:
                            self._buffer.append(next(self._iter))
                        except StopIteration:
                            if len(self._buffer) != 0:
                                while len(self._buffer) > 0:
                                    yield self._buffer.popleft()
                            self._iter = None
                            return
                while len(self._buffer) > self._prefetch_buffer_size // 2:
                    elem = self._buffer.popleft()
                    yield elem

        else:
            yield from self.dp


@functional_datapipe("repeat")
class Repeat(IterDataPipe):
    """
    Repeats the elements of the data pipe a specified number of times.

    Args:
        source_pipe:
            Source IterDataPipe to repeat.
        count:
            Number of times to repeat the dataset; -1 for indefinite repetition.
    """

    def __init__(self, source_pipe: IterDataPipe, count: int = -1):
        super().__init__()
        self.dp = source_pipe
        self.count = count

    def __iter__(self):
        if self.count == 1:
            yield from self.dp
            return
        i = 0
        while self.count < 0 or i < self.count:
            for elem in self.dp:
                new_elem = copy.copy(elem)
                yield new_elem
            i += 1


@functional_datapipe("shard")
class Shard(ShardingFilterIterDataPipe):
    """
    Applies sharding to distribute data across multiple instances.

    Args:
        source_pipe:
            Source IterDataPipe for distribution.
        partition:
            If True, data is partitioned among different instances; otherwise, all instances see all data.
    """

    def __init__(self, source_pipe: IterDataPipe, partition: bool = True):
        super().__init__(source_pipe, None)
        self.partition = partition
        self.dp = source_pipe

    def apply_sharding(
        self,
        num_of_instances: int,
        instance_id: int,
        sharding_group: SHARDING_PRIORITIES,
    ):
        if self.partition:
            return super().apply_sharding(num_of_instances, instance_id, sharding_group)
        else:
            # We can not handle uneven data for evaluation on DDP, so we don't
            # sample data by rank, that means every GPU gets the same
            # and all the evaluation data
            info = torch.utils.data.get_worker_info()
            if info is None:
                self.num_of_instances = 1
                self.instance_id = 0
            else:
                n_workers_per_device = info.num_workers
                self.num_of_instances = n_workers_per_device
                self.instance_id = info.id


class JasonLinePipe(IterDataPipe):
    """
    Streams json lines from jsonl files.

    Args:
        file_paths (Union[Path, List[Path]]):
            Path or list of paths to the jsonl files to be read.
        mode:
            File opening mode, typically 'r' for reading.
    """

    def __init__(self, file_paths: Union[Path, List[Path]], mode: str = "r"):
        super().__init__()
        file_lister = datapipes.iter.FileLister(file_paths)
        source_pipe = datapipes.iter.FileOpener(
            file_lister, mode=mode, encoding="utf-8"
        )
        self.dp = source_pipe

    def __iter__(self):
        for fname, stream in self.dp:
            try:
                for idx, line in enumerate(stream):
                    yield json.loads(line)
            except Exception as e:
                print("Error raises in the JasonLinePipe", e, stream)
                continue
            stream.close()


class RawDataset(IterDataPipe):
    """
    Represents a raw dataset, applying various transformations like shuffling and prefetching.

    Args:
        file_paths (Union[Path, List[Path]]):
            Path or list of paths to data files.
        prefetch (int):
            Number of items to prefetch, improving data retrieval performance.
        partition (bool):
            Whether to apply sharding to distribute data across multiple processing units.
        shuffle (bool):
            Enables shuffling of the data.
        shuffle_size (int):
            Buffer size used for shuffling, impacts randomness and memory usage.
        cycle (int):
            Number of times the dataset should be repeated.
    """

    def __init__(
        self,
        file_paths: Union[Path, List[Path]],
        prefetch: int = 500,
        partition: bool = True,
        shuffle: bool = False,
        shuffle_size: int = 10000,
        cycle: int = 1,
    ):
        super().__init__()
        self.dp = JasonLinePipe(file_paths).repeat(cycle).prefetch(prefetch)
        if shuffle:
            self.dp = self.dp.shuffle(buffer_size=shuffle_size)
        self.dp = self.dp.shard(partition)

    def __iter__(self):
        for d in self.dp:
            yield d


# Test
if __name__ == "__main__":
    import random
    class JasonLinePipe:
        def __init__(self, file_paths):
            self.data = [{"id": i} for i in range(100)]
        def __iter__(self):
            for d in self.data:
                yield d
        def repeat(self, cycle):
            self.data = self.data * cycle
            return self
        def prefetch(self, n):
            return self
        def shuffle(self, buffer_size=10000):
            random.shuffle(self.data)
            return self
        def shard(self, partition):
            return self

    dataset = RawDataset(
        file_paths=["dummy.jsonl"],
        shuffle=True,
        cycle=1,
    )

    dataset = dataset.shuffle(buffer_size=10000)

    for epoch in range(3):
        print(f"=== Epoch {epoch} ===")
        ids = [d["id"] for d in dataset]
        print("The first 10 sample ids:", ids[:10])

# python -m models.base.base_datapipes