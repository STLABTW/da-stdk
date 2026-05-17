"""
Tests for da_stdk/models/stdk_mlp.py.
Focuses on SpatialBasisEmbedding init variants, TemporalBasisEmbedding,
and create_model.
"""

import numpy as np
import pytest
import torch

from da_stdk.models.stdk_mlp import (
    STDKMLP,
    SpatialBasisEmbedding,
    TemporalBasisEmbedding,
    create_model,
)

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# SpatialBasisEmbedding - different init methods
# ---------------------------------------------------------------------------
class TestSpatialBasisEmbeddingInits:
    @pytest.fixture
    def train_coords(self):
        return RNG.uniform(0, 1, (200, 2)).astype(np.float32)

    def test_uniform_init(self, capsys):
        emb = SpatialBasisEmbedding(n_centers=[9, 25], init_method="uniform")
        coords = torch.rand(10, 2)
        out = emb(coords)
        assert out.shape[0] == 10

    def test_random_site_init(self, train_coords, capsys):
        emb = SpatialBasisEmbedding(
            n_centers=[9], init_method="random_site", train_coords=train_coords
        )
        coords = torch.rand(5, 2)
        out = emb(coords)
        assert out.shape[0] == 5

    def test_kmeans_balanced_init(self, train_coords, capsys):
        emb = SpatialBasisEmbedding(
            n_centers=[9], init_method="kmeans_balanced", train_coords=train_coords
        )
        coords = torch.rand(5, 2)
        out = emb(coords)
        assert out.shape[0] == 5

    def test_gmm_init(self, train_coords, capsys):
        emb = SpatialBasisEmbedding(n_centers=[5], init_method="gmm", train_coords=train_coords)
        coords = torch.rand(8, 2)
        out = emb(coords)
        assert out.shape[0] == 8

    def test_invalid_init_raises(self):
        with pytest.raises(ValueError):
            SpatialBasisEmbedding(n_centers=[9], init_method="unknown")

    def test_invalid_basis_function_raises(self):
        with pytest.raises(ValueError):
            SpatialBasisEmbedding(n_centers=[9], basis_function="unknown_fn")

    def test_learnable_true(self, capsys):
        emb = SpatialBasisEmbedding(n_centers=[9], learnable=True)
        assert emb.centers.requires_grad
        assert emb.log_bandwidths.requires_grad

    def test_gaussian_basis(self, capsys):
        emb = SpatialBasisEmbedding(n_centers=[9], basis_function="gaussian")
        coords = torch.rand(4, 2)
        out = emb(coords)
        assert out.shape[0] == 4

    def test_triangular_basis(self, capsys):
        emb = SpatialBasisEmbedding(n_centers=[9], basis_function="triangular")
        coords = torch.rand(4, 2)
        out = emb(coords)
        assert out.shape[0] == 4

    def test_gradient_damping(self, capsys):
        emb = SpatialBasisEmbedding(n_centers=[9], learnable=True, gradient_damping=True)
        assert emb.centers.requires_grad

    def test_output_nonneg_wendland(self, capsys):
        emb = SpatialBasisEmbedding(n_centers=[9], basis_function="wendland")
        coords = torch.rand(20, 2)
        out = emb(coords)
        assert torch.all(out >= 0.0)

    def test_no_train_coords_raises_for_gmm(self):
        with pytest.raises(AssertionError):
            SpatialBasisEmbedding(n_centers=[5], init_method="gmm", train_coords=None)

    def test_no_train_coords_raises_for_random_site(self):
        with pytest.raises(AssertionError):
            SpatialBasisEmbedding(n_centers=[5], init_method="random_site", train_coords=None)

    def test_no_train_coords_raises_for_kmeans(self):
        with pytest.raises(AssertionError):
            SpatialBasisEmbedding(n_centers=[5], init_method="kmeans_balanced", train_coords=None)


# ---------------------------------------------------------------------------
# TemporalBasisEmbedding
# ---------------------------------------------------------------------------
class TestTemporalBasisEmbedding:
    def test_output_shape(self):
        emb = TemporalBasisEmbedding(n_centers=[5, 10])
        t = torch.rand(8, 1)
        out = emb(t)
        assert out.shape == (8, 15)

    def test_output_in_0_1(self):
        emb = TemporalBasisEmbedding(n_centers=[5])
        t = torch.rand(20, 1)
        out = emb(t)
        assert torch.all(out >= 0.0)
        assert torch.all(out <= 1.0 + 1e-5)

    def test_single_center(self):
        emb = TemporalBasisEmbedding(n_centers=[1])
        t = torch.tensor([[0.5]])
        out = emb(t)
        assert out.shape == (1, 1)

    def test_deterministic(self):
        emb = TemporalBasisEmbedding(n_centers=[10])
        t = torch.rand(5, 1)
        assert torch.allclose(emb(t), emb(t))


# ---------------------------------------------------------------------------
# create_model
# ---------------------------------------------------------------------------
class TestCreateModel:
    def test_mean_regression(self, capsys):
        config = {
            "regression_type": "mean",
            "k_spatial_centers": [9],
            "k_temporal_centers": [5],
            "hidden_dims": [32, 16],
            "p_covariates": 0,
        }
        model = create_model(config)
        assert isinstance(model, STDKMLP)

    def test_multi_quantile_regression(self, capsys):
        config = {
            "regression_type": "multi-quantile",
            "quantile_levels": [0.1, 0.5, 0.9],
            "k_spatial_centers": [9],
            "k_temporal_centers": [5],
            "hidden_dims": [32, 16],
            "p_covariates": 0,
        }
        model = create_model(config)
        assert isinstance(model, STDKMLP)

    def test_with_train_coords(self, capsys):
        coords = RNG.uniform(0, 1, (100, 2)).astype(np.float32)
        config = {
            "regression_type": "mean",
            "spatial_init_method": "kmeans_balanced",
            "k_spatial_centers": [9],
            "k_temporal_centers": [5],
            "hidden_dims": [16],
            "p_covariates": 0,
        }
        model = create_model(config, train_coords=coords)
        assert isinstance(model, STDKMLP)

    def test_default_config(self, capsys):
        model = create_model({})
        assert isinstance(model, STDKMLP)
