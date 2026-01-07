import tensorflow as tf
import numpy as np


def get_criteo_tf_dataset(
    npz_file_path: str,
    split: str = "train",
    batch_size: int = 1024,
    shuffle: bool = True,
    seed: int = 42,
) -> tf.data.Dataset:

    print(f"Loading data from {npz_file_path} for split: {split}...")
    with np.load(npz_file_path) as data:
        raw_y = data["y"].astype(np.int32)
        raw_dense = data["X_int"].astype(np.float32)
        raw_sparse = data["X_cat"].astype(np.int32)

    total_len = raw_y.shape[0]
    assert total_len == 45840617, "The dataset size does not match expected size."

    train_size = 39291958
    if split == "train":
        labels = raw_y[:train_size]
        dense_features = raw_dense[:train_size]
        sparse_features = raw_sparse[:train_size]
    elif split == "valid":
        labels = raw_y[train_size:]
        dense_features = raw_dense[train_size:]
        sparse_features = raw_sparse[train_size:]
    else:
        raise ValueError("split must be 'train' or 'valid'")

    del raw_y, raw_dense, raw_sparse

    np.log1p(dense_features, out=dense_features)

    if shuffle and split == "train":
        print("Shuffling data in memory...")
        indices = np.arange(len(labels))
        np.random.seed(seed)
        np.random.shuffle(indices)

        labels = labels[indices]
        dense_features = dense_features[indices]
        sparse_features = sparse_features[indices]

    print("Creating tf.data.Dataset pipeline...")

    dataset = tf.data.Dataset.from_tensor_slices(
        ((sparse_features, dense_features), labels)
    )

    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset
