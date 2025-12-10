import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from astropy.io import fits

from code.config import SIXTEConfig, ModelConfig, MLConfig
sixte_config = SIXTEConfig()
model_config = ModelConfig()
ml_config = MLConfig()

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

LABEL_DICT = {"kt": r"$kT$ [keV]",
              "src_flux": r"Flux [$\mathrm{erg}\,\mathrm{cm}^{-2}\,\mathrm{s}^{-1}$]",
              "nh": r"$N_\mathrm{H}$ [$10^{22}\,\mathrm{cm}^{-2}$]"}

def plot_loss(metadata, label = "Loss"):
    fig, axes = plt.subplots(ncols=1, nrows=2, sharex=True, figsize=(12/2.54, 12/2.54))
    # Note that running training loss (not plotted here) should start half an epoch
    # earlier (at 0.5) because it is calculated during training while the gradients
    # are still updated.
    axes[0].plot(np.arange(len(metadata.training_losses))+1, metadata.training_losses, label="Training Loss", color="red")
    axes[0].plot(np.arange(len(metadata.validation_losses))+1, metadata.validation_losses, label="Validation Loss", color="blue")

    axes[0].set_ylabel(label)
    axes[0].legend()

    axes[-1].plot(np.arange(len(metadata.mean_total_grad_norms))+1, metadata.mean_total_grad_norms)
    axes[-1].set_yscale('log')
    axes[-1].set_ylabel("Mean Gradient Norms")

    axes[-1].set_xlabel('Epoch')
    plt.tight_layout()
    outfile = "loss.pdf"
    print("Wrote to {}".format(outfile))
    plt.savefig(outfile)

def _setup_plot(plot_input_data_only = False):
    if plot_input_data_only == True:
        fig, axes = plt.subplots(1, 1, figsize=(10 / 2.54, 7 / 2.54), dpi=300)
        axes.set_ylabel('Counts')
        axes.set_xscale('log')
        axes.set_yscale('log')
        axes.set_xlim(20, 500)
    else:
        fig, axes = plt.subplots(2, 1, figsize=(10/2.54, 7/2.54), dpi=300,
                                 sharex=True, gridspec_kw={'height_ratios': [3, 1], 'hspace': 0})
        axes[0].set_ylabel('Counts')
        axes[0].set_xscale('log')
        axes[0].set_yscale('log')
        axes[0].set_xlim(20, 500)

        axes[1].axhline(y=1, color='gray', linestyle='--', linewidth=1)  # Reference line at ratio = 1
        axes[1].set_xlabel('Channel')
        axes[1].set_xscale('log')
        axes[1].set_xlim(20, 500)

    return fig, axes

def _plot_ratio(axis, data1, data2, data1_label = "Target", data2_label = "predicted"):
    ratio = torch.where(data2 != 0, data1 / data2, torch.nan)
    axis.plot(ratio, color='black')
    axis.set_ylabel(fr'$\frac{{\text{{{data1_label}}}}}{{\text{{{data2_label}}}}}$')
    axis.set_ylim(0, 2)
    return axis

def unite_pdfs(outfiles, outfilename = "testdata.pdf", remove = True):
    outstr = " ".join(outfiles)
    os.system(f"pdfunite {outstr} {outfilename}")
    print(f"Wrote {outfilename}")
    if remove:
        [os.system(f"rm {fp}") for fp in outfiles]

def _write_pha_file(channels, predicted_output, output_fname):
    de_piledup_spectrum = predicted_output.numpy()  # .astype(count_spectrum.dtype)

    primary_hdu = fits.PrimaryHDU()

    header = fits.Header()
    header["BACKFILE"] = "NONE"
    header["RESPFILE"] = sixte_config.MASTER_RMF
    header["ANCRFILE"] = sixte_config.MASTER_ARF

    columns = fits.ColDefs([
        fits.Column(name="CHANNEL", format="J", array=channels),
        fits.Column(name="COUNTS", format="J", array=de_piledup_spectrum)
    ])
    table_hdu = fits.BinTableHDU.from_columns(columns, header=header, name="SPECTRUM")

    hdulist = fits.HDUList([primary_hdu, table_hdu])
    hdulist.writeto(output_fname, overwrite=True)

def evaluate_on_test_spectrum(model, test_dataset, phafile = False, plot_input_data_only = False):
    if phafile == False:
        indices = [int(np.random.uniform(0, len(test_dataset)-1)) for _ in range(20)]
    else:
        indices = [test_dataset.index_of_input(phafile)]
        
    outfiles = []

    for index in indices:
        input_data, target_data = test_dataset[index]
        input_fname, target_fname = test_dataset.get_filenames(index)

        fix, axes = _setup_plot(plot_input_data_only = plot_input_data_only)
        axes.set_title(fr"\small {os.path.basename(input_fname)}")
        axes.plot(input_data[0, :], label=r"Input 0--30 arcsec", linewidth=1)
        axes.plot(input_data[1, :], label=r"Input 30--60 arcsec", linewidth=1)
        axes.plot(input_data[2, :], label=r"Input 60--120 arcsec", linewidth=1)
        axes.plot(input_data[3, :], label=r"Input 120--240 arcsec", linewidth=1)

        if plot_input_data_only == False:
            model.eval()
            with torch.no_grad():  # Disable gradient calculation for inference
                predicted_output = model(input_data.unsqueeze(0))[0]

            # Bug in normalization! Hack: rescale for now
            predicted_output *= max(input_data) / max(predicted_output)
            target_data *= max(input_data) / max(target_data)
            ymax = max(predicted_output)

            axes[0].plot(predicted_output, label="Predicted (rescaled)", linewidth=1)
            axes[0].plot(target_data, label="Target (rescaled)", linewidth=1)
            axes[0].set_ylim(0.7, ymax + 0.1 * ymax)

            axes[1] = _plot_ratio(axes[1], target_data, predicted_output, data1_label="Target")
            axes[0].legend()
        else:
            axes.legend()

        outfile = f"outfiles/testdata_{index}.pdf"
        outfiles.append(outfile)
        plt.tight_layout()
        plt.savefig(outfile)

    if phafile == False:
        unite_pdfs(outfiles)
    else:
        print("Wrote {}".format(*outfiles))

def plot_parameter_distributions(targets, title=""):
    fig, axes = plt.subplots(1, ml_config.dim_output_parameters, figsize=(15, 4), sharey=True)

    colors = ['blue', 'green', 'red']
    nbins = 50

    for ii in range(ml_config.dim_output_parameters):
        axes[ii].hist(targets[:, ii], bins=nbins, alpha=0.7, color=colors[ii], edgecolor='black')
        axes[ii].set_xlabel(model_config.names[ii])

    axes[0].set_ylabel('Number of samples')
    fig.suptitle(f"{title} dataset ({len(targets)} samples)")
    plt.subplots_adjust(wspace=0)
    plt.tight_layout()
    plt.savefig(f"{title}_dataset.pdf")
    print("Wrote {}_dataset.pdf".format(title))

def print_dataset_statistics(targets, title="Dataset"):
    print(f"\n{title} statistics:")
    print("=" * 50)
    print(f"Number of samples: {len(targets)}")

    for ii, name in enumerate(model_config.names):
        values = targets[:, ii]
        print(f"\n{name}")
        print(f"\tRange: {values.min()} - {values.max()}")
        print(f"\tMean: {values.mean()}, Stddev: {values.std()}")
        print(f"\tMedian: {values.median()}")
