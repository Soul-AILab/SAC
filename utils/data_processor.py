def feats_length_fn(sample: dict, length_key: str):
    """Retrieve the length of features from a sample based on the specified key.

    Args:
        sample (dict):
            A dictionary containing various data points including the length.
        length_key (str):
            The key corresponding to the length value in the sample.

    Returns:
        int: The length of the sample as specified by `length_key`.
    """
    assert length_key in sample, f"Key '{length_key}' not found in sample."

    return sample[length_key]


class DynamicBatchWindow:
    """A class to determine if a new batch should be started based on the maximum number of frames allowed.

    Args:
        max_frames_in_batch (int): The maximum number of frames a batch can hold.
        length_key (str): The key used to retrieve the length of a sample.
    """

    def __init__(self, max_frames_in_batch: int, length_key: str):
        self.longest_frames = 0
        self.length_key = length_key
        self.max_frames_in_batch = max_frames_in_batch

    def __call__(self, sample: dict, buffer_size: int) -> bool:
        """Determine if adding another sample exceeds the maximum frame limit for a batch.

        Args:
            sample (dict): The current sample to be added to the batch.
            buffer_size (int): The current size of the buffer (number of samples).

        Returns:
            bool: True if adding the sample exceeds the max limit and a new batch should start, False otherwise.

        Raises:
            AssertionError: If the sample does not contain the expected `length_key`.
        """
        assert isinstance(sample, dict), "Expected sample to be a dictionary."
        assert (
            self.length_key in sample
        ), f"Key '{self.length_key}' not found in sample."

        new_sample_frames = sample[self.length_key]
        self.longest_frames = max(self.longest_frames, new_sample_frames)
        frames_after_padding = self.longest_frames * (buffer_size + 1)

        if frames_after_padding > self.max_frames_in_batch:
            self.longest_frames = (
                new_sample_frames  # Reset longest_frames for the next batch
            )
            return True
        return False