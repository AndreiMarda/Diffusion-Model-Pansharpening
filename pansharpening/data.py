from bisect import bisect_right
from pathlib import Path

import h5py
import torch
from torch.utils.data import DataLoader, Dataset


class PanCollectionDataset(Dataset):

    def __init__(
        self,
        h5_paths,
        has_gt=True,
        normalize=True,
        max_value=1023.0,
        augment=False,
    ):
        if isinstance(h5_paths, (str, Path)):
            h5_paths = [h5_paths]

        self.h5_paths = [Path(path) for path in h5_paths]
        self.has_gt = has_gt
        self.normalize = normalize
        self.max_value = float(max_value)
        self.augment = augment
        if not self.h5_paths:
            raise ValueError("At least one H5 file is required.")

        self.lengths = []
        self.cumulative_lengths = []
        self._files = {}
        total = 0
        channel_counts = set()

        for h5_path in self.h5_paths:
            if not h5_path.exists():
                raise FileNotFoundError(f"H5 file not found: {h5_path}")

            with h5py.File(h5_path, "r") as f:
                required_keys = ["pan", "ms"]
                if self.has_gt:
                    required_keys.append("gt")

                missing = [key for key in required_keys if key not in f]
                if missing:
                    raise KeyError(f"{h5_path} is missing keys: {missing}")

                length = f["pan"].shape[0]
                total += length
                self.lengths.append(length)
                self.cumulative_lengths.append(total)
                channel_counts.add(f["ms"].shape[1])

        if len(channel_counts) > 1:
            raise ValueError(
                "Cannot batch H5 files with different channel counts: "
                f"{sorted(channel_counts)}"
            )

    def __len__(self):
        return self.cumulative_lengths[-1]

    def _locate(self, index):
        file_index = bisect_right(self.cumulative_lengths, index)
        previous = 0 if file_index == 0 else self.cumulative_lengths[file_index - 1]
        return file_index, index - previous

    def _file(self, file_index):
        if file_index not in self._files:
            self._files[file_index] = h5py.File(self.h5_paths[file_index], "r")
        return self._files[file_index]

    def __getitem__(self, index):
        file_index, local_index = self._locate(index)
        f = self._file(file_index)

        sample = {
            "pan": torch.from_numpy(f["pan"][local_index]).float(),
            "ms": torch.from_numpy(f["ms"][local_index]).float(),
        }

        if self.has_gt and "gt" in f:
            sample["gt"] = torch.from_numpy(f["gt"][local_index]).float()

        if "lms" in f:
            sample["lms"] = torch.from_numpy(f["lms"][local_index]).float()

        if self.augment:
            sample = self.augment_sample(sample)

        if self.normalize:
            for key in sample:
                sample[key] = self.normalize_image(sample[key])

        return sample

    def close(self):
        for h5_file in self._files.values():
            h5_file.close()
        self._files.clear()

    def normalize_image(self, x):
        x = x.float() / self.max_value
        x = x.clamp(0.0, 1.0)
        x = 2.0 * x - 1.0
        return x

    def augment_sample(self, sample):
        # Apply the same geometric transform to PAN, MS, LMS, and GT.
        if torch.rand(()) < 0.5:
            sample = {
                key: torch.flip(value, dims=(-1,))
                for key, value in sample.items()
            }

        if torch.rand(()) < 0.5:
            sample = {
                key: torch.flip(value, dims=(-2,))
                for key, value in sample.items()
            }

        can_rotate_90 = all(
            value.shape[-2] == value.shape[-1]
            for value in sample.values()
        )
        if can_rotate_90:
            k = int(torch.randint(0, 4, ()).item())
        else:
            k = int(torch.randint(0, 2, ()).item()) * 2

        if k:
            sample = {
                key: torch.rot90(value, k=k, dims=(-2, -1))
                for key, value in sample.items()
            }

        return sample


def get_split_h5_paths(split_dir, dataset_name="gf2", prefix="train"):
    split_dir = Path(split_dir)

    if dataset_name is None:
        return sorted(split_dir.glob(f"{prefix}_*.h5"))

    h5_path = split_dir / f"{prefix}_{dataset_name}.h5"
    if not h5_path.exists():
        available = ", ".join(path.stem.replace(f"{prefix}_", "") for path in sorted(split_dir.glob(f"{prefix}_*.h5")))
        raise FileNotFoundError(
            f"H5 file not found: {h5_path}. Available datasets: {available}"
        )

    return [h5_path]


def get_training_h5_paths(training_dir="dataset/training", dataset_name="gf2"):
    return get_split_h5_paths(training_dir, dataset_name, prefix="train")


def get_validation_h5_paths(validation_dir="dataset/validation", dataset_name="gf2"):
    return get_split_h5_paths(validation_dir, dataset_name, prefix="valid")


def create_loader(
    h5_paths,
    batch_size=8,
    shuffle=True,
    drop_last=True,
    pin_memory=None,
    has_gt=True,
    normalize=True,
    max_value=1023.0,
    augment=False,
):
    dataset = PanCollectionDataset(
        h5_paths,
        has_gt=has_gt,
        normalize=normalize,
        max_value=max_value,
        augment=augment,
    )

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        pin_memory=pin_memory,
    )


def create_train_loader(
    training_dir="dataset/training",
    dataset_name="gf2",
    batch_size=8,
    shuffle=True,
    drop_last=True,
    pin_memory=None,
    normalize=True,
    max_value=1023.0,
    augment=True,
):
    h5_paths = get_training_h5_paths(training_dir, dataset_name)
    return create_loader(
        h5_paths=h5_paths,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        pin_memory=pin_memory,
        normalize=normalize,
        max_value=max_value,
        augment=augment,
    )


def create_validation_loader(
    validation_dir="dataset/validation",
    dataset_name="gf2",
    batch_size=8,
    shuffle=False,
    drop_last=False,
    pin_memory=None,
    normalize=True,
    max_value=1023.0,
):
    h5_paths = get_validation_h5_paths(validation_dir, dataset_name)
    return create_loader(
        h5_paths=h5_paths,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        pin_memory=pin_memory,
        normalize=normalize,
        max_value=max_value,
    )


def get_test_h5_paths(testing_dir="dataset/testing", dataset_name="gf2"):
    full_dir = Path(testing_dir) / dataset_name / "Full"
    h5_paths = sorted(full_dir.glob("*.h5"))
    if not h5_paths:
        raise FileNotFoundError(f"No Full H5 test files found in {full_dir}")
    return h5_paths


def create_test_loader(
    testing_dir="dataset/testing",
    dataset_name="gf2",
    batch_size=1,
    shuffle=False,
    pin_memory=None,
    normalize=True,
    max_value=1023.0,
):
    h5_paths = get_test_h5_paths(testing_dir, dataset_name)
    return create_loader(
        h5_paths=h5_paths,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        pin_memory=pin_memory,
        normalize=normalize,
        max_value=max_value,
        has_gt=False,
    )
