import glob
import re
import torch
from astropy.io import fits
from torch.utils.data import Dataset, Subset, random_split
import numpy as np
from sklearn.preprocessing import StandardScaler

from subs import print_dataset_statistics, plot_parameter_distributions
from config import SIXTEConfig, MLConfig

sixte_config = SIXTEConfig()
ml_config = MLConfig()

class PileupDataset(Dataset):
    def __init__(self, input_files, target_files = None, scaler = None, fit_scaler = None):
        self.input_files = input_files
        self.target_files = target_files if target_files else [None] * len(input_files)
        self.scaler = scaler

        if fit_scaler:
            self.fit_scaler()

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        # for using four spatially resolved spectra as input
        input_data_circle0 = fits.getdata(self.input_files[idx])
        input_data_annulus1 = fits.getdata(self.input_files[idx].replace('circle0', 'annulus1'))
        input_data_annulus2 = fits.getdata(self.input_files[idx].replace('circle0', 'annulus2'))
        input_data_annulus3 = fits.getdata(self.input_files[idx].replace('circle0', 'annulus3'))

        input_counts_circle0 = np.array(input_data_circle0["COUNTS"], dtype=np.float32)
        input_counts_annulus1 = np.array(input_data_annulus1["COUNTS"], dtype=np.float32)
        input_counts_annulus2 = np.array(input_data_annulus2["COUNTS"], dtype=np.float32)
        input_counts_annulus3 = np.array(input_data_annulus3["COUNTS"], dtype=np.float32)
        counts = [torch.from_numpy(input_counts_circle0),
                  torch.from_numpy(input_counts_annulus1),
                  torch.from_numpy(input_counts_annulus2),
                  torch.from_numpy(input_counts_annulus3)]

        input_tensor = torch.stack(counts, dim=0)

        (kt, src_flux, nh) = self._get_targets_from_fits_file(self.input_files[idx])
        target_tensor = torch.tensor([kt, src_flux, nh])

        if self.scaler:
            # Need to reshape from (3,), so from 1D array, to (1, 3) because transform expects 2D array.
            # flatten transforms back to (3,).
            target_normalized = self.scaler.transform(target_tensor.reshape(1, -1)).flatten()
            target_tensor = torch.from_numpy(target_normalized).float()

        return input_tensor, target_tensor

    def __TEST1__getitem__(self, idx):
        input_data = fits.getdata(self.input_files[idx])
        input_counts = np.array(input_data["COUNTS"], dtype=np.float32)
        input_tensor = torch.tensor(input_counts)

        (kt, src_flux, nh) = self._get_targets_from_fits_file(self.input_files[idx])

        target_tensor = torch.tensor([kt, src_flux, nh])

        return input_tensor, target_tensor

    def __TEST2__getitem__(self, idx):
        # For regression to the unpiled spectrum
        input_data = fits.getdata(self.input_files[idx])
        target_data = fits.getdata(self.target_files[idx])

        input_counts = np.array(input_data["COUNTS"], dtype=np.float32)
        target_counts = np.array(target_data["COUNTS"], dtype=np.float32)

        input_tensor = torch.tensor(input_counts)
        target_tensor = torch.tensor(target_counts)

        return input_tensor, target_tensor

    def get_filenames(self, idx):
        return self.input_files[idx], self.target_files[idx]

    def _get_targets_from_fits_file(self, filename):
        src_flux = fits.getval(filename, "SRC_FLUX", ext=1)

        # Transform of log grid to linear range, otherwise standardization biased towards high fluxes
        src_flux = np.log10(src_flux)

        kt = fits.getval(filename, "KT", ext=1)
        nh = fits.getval(filename, "NH", ext=1)

        return kt, src_flux, nh

    def fit_scaler(self):
        all_targets = []
        for filename in self.input_files:
            (kt, src_flux, nh) = self._get_targets_from_fits_file(filename)
            all_targets.append(np.array([kt, src_flux, nh]))

        self.scaler = StandardScaler().fit(np.array(all_targets))


class SubsetWithFilenames(Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        # Create look-up table for file indices after random_split
        self.filenames = [(dataset.input_files[i], dataset.target_files[i]) for i in self.indices]
        self.index_by_input  = {inp: i for i, (inp, _) in enumerate(self.filenames)}
        self.index_by_target = {tgt: i for i, (_, tgt) in enumerate(self.filenames)}
        
    def get_filenames(self, idx):
        return self.dataset.get_filenames(self.indices[idx])

    def index_of_input(self, path: str) -> int:
        return self.index_by_input[path]


def get_src_flux_from_filename(fname):
    src_flux = re.findall("[0-9]+.[0-9]+Em10", fname)[0]
    return float(src_flux.split("Em10")[0]) * 1e-10

def load_and_split_dataset():
    piledup = glob.glob(sixte_config.SPECDIR + "*cgs*piledup_circle0.fits")

    torch.manual_seed(ml_config.dataloader_random_seed)

    train_size = int(0.7 * len(piledup))
    val_size = int(0.15 * len(piledup))
    test_size = len(piledup) - train_size - val_size  # Ensure all samples are used

    train_files, val_files, test_files = random_split(piledup, [train_size, val_size, test_size])

    train_dataset = PileupDataset(train_files, fit_scaler = True)
    val_dataset = PileupDataset(val_files, scaler = train_dataset.scaler)
    test_dataset = PileupDataset(test_files, scaler = train_dataset.scaler)

    return train_dataset, val_dataset, test_dataset

def main():
    def test_grid():
        train_dataset, val_dataset, test_dataset = load_and_split_dataset()
        targets = torch.stack([train_dataset[ii][1] for ii in range(len(train_dataset))])
        print_dataset_statistics(targets, title="Training")
        plot_parameter_distributions(targets, title="Training")

    def test2():
        piledup = glob.glob(sixte_config.SPECDIR + "*cgs*.fits")
        nonpiledup = [pha.replace("piledup", "nonpiledup") for pha in piledup]
        torch.manual_seed(ml_config.dataloader_random_seed)
        dataset = PileupDataset(piledup, target_files=nonpiledup)
        print(dataset.__getitem__(0))

    def test_filename_indexing():
        train_dataset, val_dataset, test_dataset = load_and_split_dataset()
        input_name = "/pool/burg1/astroai/pileup/sims/spec_piledup/1.223821Em10cgs_0.981670nH_0.139405kT_piledup_circle0.fits"
        idx = test_dataset.index_of_input(input_name)
        (inp, target) = test_dataset[idx]

    test_grid()

if __name__ == "__main__":
    main()
