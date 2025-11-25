import pytest
import quantom_ips.envs.parsers.multi_numpy_parser
from quantom_ips.utils import make
from omegaconf import OmegaConf
import numpy as np
import os
import shutil


class TestMultiNumpyParser:

    # Create the test data:
    def setup_method(self, test_method):
        print("Create test data")
        self.n_data_sets = 5
        self.n_files_per_set = 4
        self.n_samples = 1000
        self.dim = 3
        self.data_loc = "multi_numpy_parser_test_data"
        self.data_paths = []

        os.makedirs(self.data_loc, exist_ok=True)
        # ++++++++++++++++++++
        for s in range(self.n_data_sets):
            current_paths = []
            # ++++++++++++++++++++
            for p in range(self.n_files_per_set):
                current_loc = f"{self.data_loc}/set{s}_file{p}.npy"
                current_paths.append(current_loc)
                current_data = np.random.normal(size=(self.n_samples, self.dim))
                np.save(current_loc, current_data)
            # ++++++++++++++++++++
            self.data_paths.append(current_paths)
        # ++++++++++++++++++++

    # Delete the data set if not needed anymore:
    def teardown_method(self, test_method):
        print("Delete test data")
        if os.path.exists(self.data_loc):
            shutil.rmtree(self.data_loc)

    # Run test, using all the data:
    def test_with_all_data(self):
        print("Run test with all data")

        # Get the module:
        data_fraction = 0.5
        cfg = {
            "paths": self.data_paths,
            "event_axis": 0,
            "data_fraction": data_fraction,
            "data_size": -1,
            "n_randomly_chosen_sets": -1,
            "chosen_data_set_idx":[]
        }
        cfg = OmegaConf.create(cfg)
        parser = make("MultiNumpyParser", config=cfg, devices=None)

        # Retreive data:
        data = parser.data

        # Sanity checks
        assert data.shape[0] == self.n_data_sets
        assert data.shape[1] == int(
            data_fraction * self.n_samples * self.n_files_per_set
        )

        # Now take samples from that data:
        batch_size = 30
        n_particles = 2
        n_samples = 5
        sampled_data = parser.get_samples(
            sample_shape=(batch_size, n_particles, n_samples, self.dim)
        )

        # Final sanity checks
        assert sampled_data.shape[0] == self.n_data_sets
        assert sampled_data.shape[1] == batch_size
        assert sampled_data.shape[2] == n_particles
        assert sampled_data.shape[3] == n_samples
        assert sampled_data.shape[4] == self.dim

    # Run a test, with selection only n_sub randomly chosen subsets of the available data sets:
    def test_with_random_subset_data(self):
        print("Run test with random subset data")

        # Get the module:
        n_subsets = 3
        data_fraction = 0.5
        cfg = {
            "paths": self.data_paths,
            "event_axis": 0,
            "data_fraction": data_fraction,
            "data_size": -1,
            "n_randomly_chosen_sets": n_subsets,
            "chosen_data_set_idx":[]
        }
        cfg = OmegaConf.create(cfg)
        parser = make("MultiNumpyParser", config=cfg, devices=None)

        # Retreive data:
        data = parser.data

        # Sanity checks
        if n_subsets > 1:
            assert data.shape[0] == n_subsets
            assert data.shape[1] == int(
                data_fraction * self.n_samples * self.n_files_per_set
            )
        else:
            assert data.shape[0] == int(
                data_fraction * self.n_samples * self.n_files_per_set
            )

        # Now take samples from that data:
        batch_size = 30
        n_particles = 2
        n_samples = 5
        sampled_data = parser.get_samples(
            sample_shape=(batch_size, n_particles, n_samples, self.dim)
        )

        # Final sanity checks
        if n_subsets > 1:
            assert sampled_data.shape[0] == n_subsets
            assert sampled_data.shape[1] == batch_size
            assert sampled_data.shape[2] == n_particles
            assert sampled_data.shape[3] == n_samples
            assert sampled_data.shape[4] == self.dim
        else:
            assert sampled_data.shape[0] == batch_size
            assert sampled_data.shape[1] == n_particles
            assert sampled_data.shape[2] == n_samples
            assert sampled_data.shape[3] == self.dim

    # Run test with a fixed chosen subset, i.e. the user asks for a specific subset:
    def test_with_fixed_subset_data(self):
        print("Run test with fixed subset data")

        # Get the module:
        fixed_set_idx = [1,2,3]
        n_subsets = len(fixed_set_idx)
        data_fraction = 0.5
        cfg = {
            "paths": self.data_paths,
            "event_axis": 0,
            "data_fraction": data_fraction,
            "data_size": -1,
            "n_randomly_chosen_sets": -1,
            "chosen_data_set_idx":fixed_set_idx
        }
        cfg = OmegaConf.create(cfg)
        parser = make("MultiNumpyParser", config=cfg, devices=None)

        # Retreive data:
        data = parser.data

        # Sanity checks
        if n_subsets > 1:
            assert data.shape[0] == n_subsets
            assert data.shape[1] == int(
                data_fraction * self.n_samples * self.n_files_per_set
            )
        else:
            assert data.shape[0] == int(
                data_fraction * self.n_samples * self.n_files_per_set
            )

        # Now take samples from that data:
        batch_size = 30
        n_particles = 2
        n_samples = 5
        sampled_data = parser.get_samples(
            sample_shape=(batch_size, n_particles, n_samples, self.dim)
        )

        # Final sanity checks
        if n_subsets > 1:
            assert sampled_data.shape[0] == n_subsets
            assert sampled_data.shape[1] == batch_size
            assert sampled_data.shape[2] == n_particles
            assert sampled_data.shape[3] == n_samples
            assert sampled_data.shape[4] == self.dim
        else:
            assert sampled_data.shape[0] == batch_size
            assert sampled_data.shape[1] == n_particles
            assert sampled_data.shape[2] == n_samples
            assert sampled_data.shape[3] == self.dim

