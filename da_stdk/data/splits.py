"""Split observed (t, s) pairs into train and validation masks."""

import numpy as np


def split_train_valid(obs_mask, obs_sites, split_method="site-wise", train_ratio=0.8, seed=None):
    """Return ``train_mask`` and ``valid_mask`` over observed (t, s) pairs."""
    if seed is not None:
        np.random.seed(seed)

    T, S = obs_mask.shape

    if split_method == "site-wise":
        n_train_sites = int(len(obs_sites) * train_ratio)
        shuffled_sites = obs_sites.copy()
        np.random.shuffle(shuffled_sites)
        train_sites = shuffled_sites[:n_train_sites]
        valid_sites = shuffled_sites[n_train_sites:]
        train_mask = np.zeros((T, S), dtype=bool)
        valid_mask = np.zeros((T, S), dtype=bool)
        train_mask[:, train_sites] = obs_mask[:, train_sites]
        valid_mask[:, valid_sites] = obs_mask[:, valid_sites]
        return train_mask, valid_mask

    if split_method == "random":
        obs_indices = np.argwhere(obs_mask)
        n_obs = len(obs_indices)
        n_train = int(n_obs * train_ratio)
        shuffled_idx = np.random.permutation(n_obs)
        train_idx = shuffled_idx[:n_train]
        valid_idx = shuffled_idx[n_train:]
        train_mask = np.zeros((T, S), dtype=bool)
        valid_mask = np.zeros((T, S), dtype=bool)
        for idx in train_idx:
            t, s = obs_indices[idx]
            train_mask[t, s] = True
        for idx in valid_idx:
            t, s = obs_indices[idx]
            valid_mask[t, s] = True
        return train_mask, valid_mask

    raise ValueError(f"Unknown split_method: {split_method}")


__all__ = ["split_train_valid"]
