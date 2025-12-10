import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import corner
import torch
from astropy.io import fits
import numpy as np
import pandas as pd
import os
import random
from typing import Tuple

from data import load_and_split_dataset
from normalizing_flow import ConvSpectraFlow
from config import MLConfig, SIXTEConfig
from subs import *

ml_config = MLConfig()
sixte_config = SIXTEConfig()

def load_emcee_fits(path, burn_in=0, thin=1, flux_factor=1.0):
    with fits.open(path) as hdul:
        rec = hdul[2].data
        nh   = np.asarray(rec["CHAINS1"], dtype=float)
        flux = np.asarray(rec["CHAINS2"], dtype=float)
        kt   = np.asarray(rec["CHAINS3"], dtype=float)

    sl = slice(burn_in, None, thin)
    nh, flux, kt = nh[sl], flux[sl], kt[sl]

    flux *= float(flux_factor)
    return np.column_stack([kt, nh])

def plot_2d_posteriors_of_spectrum(input_name, model, dataset, num_samples=10000, device='cpu'):
    model.eval()

    idx = dataset.index_of_input(input_name)

    x, y_true = dataset[idx]
    x = x.unsqueeze(0).to(device)  # add batch‐dim
    
    input_fname, target_fname = dataset.get_filenames(idx)
    
    with torch.no_grad():
        q_dist = model(x)
        samples = q_dist.sample((num_samples,))
        
    samples = samples.squeeze(1).cpu().numpy()  # remove batch dimension
    samples = np.exp(samples)  # re-transform target from log space to actual values
    
    samples[:, 1] *= ml_config.flux_factor
    y_true[1] *= ml_config.flux_factor

    names = [LABEL_DICT["kt"], LABEL_DICT["nh"]]
    ranges = [(0.123, 0.152), (0.85, 1.26)]
    # ranges = [(0.09, 0.18), (0.48, 1.6)]
    nbins = 100
    color = "red" # "green"
    basename = os.path.basename(input_name).split("_circle0.fits")[0] + "_annulus3" 
    
    mcmc = load_emcee_fits("/pool/burg1/astroai/mcmc/emcee-chain_" + basename + ".fits",
                           burn_in=1000,
                           thin=1, flux_factor=ml_config.flux_factor)
    figure = corner.corner(mcmc,
                           labels=names, bins=nbins,
                           range=ranges,
                           color=color,
                           plot_datapoints=True,
                           hist_kwargs = {'density': True, 'color': color},
                           data_kwargs = {'alpha': 0.01})


    samples = samples[:,[0,2]]
    y_true = y_true[[0, 2]]

    corner.corner(samples,
                  fig=figure,
                  labels=names,
                  color="black",
                  show_titles=False,
                  bins=50,
                  truths=y_true, truth_color='blue',
                  range=ranges,
                  hist_kwargs = {'density': True, 'color': 'black'},
                  data_kwargs = {'alpha': 0.5})

    figure.suptitle(os.path.basename(input_fname))


    handles = [Patch(color=color, label="MCMC (core-excised)"),
               # Patch(color=color, label="MCMC (all data)"),
               Patch(color='black', label="NF posterior"),
               Patch(color='blue', label="Ground truth")]

    axes = np.array(figure.axes).reshape((2, 2))
    axes[1, 0].legend(handles=handles, loc='upper right')
    
    plt.tight_layout()
    outfile = "cmp_flow_to_mcmc_" + basename + ".pdf"
    plt.savefig(outfile)
    print(f"Wrote {outfile}")


def main():
    train_dataset, val_dataset, test_dataset = load_and_split_dataset()
    model = ConvSpectraFlow()
    model.load_state_dict(torch.load(ml_config.data_neural_network + "model_weights.pth", map_location="cpu"))
    phafile = "/pool/burg1/astroai/pileup/sims/spec_piledup/1.223821Em10cgs_0.981670nH_0.139405kT_piledup_circle0.fits"
    # phafile = "/pool/burg1/astroai/pileup/sims/spec_piledup/0.020043Em10cgs_0.971797nH_0.130109kT_piledup_circle0.fits"
    
    plot_2d_posteriors_of_spectrum(phafile, model, test_dataset)
    
if __name__ == "__main__":
    main()
