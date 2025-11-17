import corner
import pandas as pd
import random
from typing import Tuple
import joblib

from data import load_and_split_dataset
from neuralnetwork import ConvSpectraFlow
from subs import *

ml_config = MLConfig()
sixte_config = SIXTEConfig()

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'


def evaluate_on_real_spectrum(model, real_pha_filename, out_pha_file = None):
    with fits.open(real_pha_filename) as hdulist:
        channels = hdulist[1].data["CHANNEL"]
        counts = hdulist[1].data["COUNTS"]

        real_dataset = torch.tensor(np.array(counts, dtype=np.float32))

        model.eval()
        with torch.no_grad():  # Disable gradient calculation for inference
            predicted_spectrum = model(real_dataset.unsqueeze(0))
            print(np.exp(predicted_spectrum))

        # fix, axes = _setup_plot()
        # axes[0].plot(real_dataset, label="eROSITA spectrum")
        # axes[0].plot(predicted_spectrum, label="Predicted by NN")
        # axes[0].legend()
        # axes[1] = _plot_ratio(axes[1], real_dataset, predicted_spectrum, data1_label="Real", data2_label="Predicted")
        # plt.tight_layout()
        # plt.show()

        # if out_pha_file is not None:
        #     _write_pha_file(channels, predicted_spectrum, out_pha_file)

def _get_params_from_output(output: torch.Tensor) -> Tuple[float, float, float, float, float, float]:
    """Split parameter means (first numbers) and log variance (last half of numbers).
    It does not quite correspond to the variance because I'm applying softplus
    instead of exp on the logarithm of the variance for numerical stability."""
    mu = output[: ml_config.dim_output_parameters]
    log_kt, log_flux, log_nh = mu.tolist()

    kt = np.exp(log_kt)  # [keV]
    flux = np.exp(log_flux) * 1e-12
    nh = np.exp(log_nh)

    raw_log_var = output[ml_config.dim_output_parameters:]
    var = torch.nn.functional.softplus(raw_log_var)
    # (do not exponentiate errors because softplus was already applied)
    kt_err, flux_err, nh_err = torch.sqrt(var).tolist()
    flux_err *= 1e-12

    return kt, flux, nh, kt_err, flux_err, nh_err

def evaluate_parameter_prediction(model, test_dataset):
    model.eval()

    kt_true, flux_true, nh_true = [], [], []
    kt_pred, flux_pred, nh_pred = [], [], []
    kt_errs, flux_errs, nh_errs = [], [], []

    with torch.no_grad():
        for input, target in test_dataset:
            # output = model(input) # .squeeze(0)
            output = model(input.unsqueeze(0))[0]  # from [1024] to [1,1024] (for convolutional layer)

            kt, flux, nh, kt_e, flux_e, nh_e = _get_params_from_output(output)
            # kt, flux, nh = output.tolist()

            kt_pred.append(kt)
            flux_pred.append(flux)
            nh_pred.append(nh)

            kt_errs.append(kt_e)
            flux_errs.append(flux_e)
            nh_errs.append(nh_e)

            kt_true.append(target[0].item())
            flux_true.append(target[1].item() * 1e-12)
            nh_true.append(target[2].item())

    kt_true, flux_true, nh_true = np.array(kt_true), np.array(flux_true), np.array(nh_true)
    kt_pred, flux_pred, nh_pred = np.array(kt_pred), np.array(flux_pred), np.array(nh_pred)
    kt_errs, flux_errs, nh_errs = np.array(kt_errs), np.array(flux_errs), np.array(nh_errs)

    fig, axes = plt.subplots(nrows = 2, ncols = 3, figsize=[6.5*3/2.54, 6*2/2.54])

    axes[0, 0].errorbar(kt_true, kt_pred, yerr=kt_errs, alpha=0.1, ms=2, ecolor="gray", elinewidth=1, fmt=".")
    axes[0, 1].errorbar(flux_true, flux_pred, yerr=flux_errs, alpha=0.1, ms=2, ecolor="gray", elinewidth=1, fmt=".")
    axes[0, 2].errorbar(nh_true, nh_pred, yerr=nh_errs, alpha=0.1, ms=2, ecolor="gray", elinewidth=1, fmt=".")
    axes[1, 0].errorbar(flux_true, (kt_pred - kt_true)/kt_true, yerr=kt_errs/kt_true, alpha=0.1, ms=2, ecolor="gray", elinewidth=1, fmt=".")
    axes[1, 1].errorbar(flux_true, (flux_pred - flux_true)/flux_true, yerr=flux_errs/flux_true, alpha=0.1, ms=2, ecolor="gray", elinewidth=1, fmt=".")
    axes[1, 2].errorbar(flux_true, (nh_pred - nh_true) / nh_true, yerr=nh_errs / nh_true, alpha=0.1, ms=2,
                        ecolor="gray", elinewidth=1, fmt=".")

    axes[0, 0].axline((0, 0), slope=1, color="gray", linestyle="--", linewidth=1, label="Ground truth")
    axes[0, 1].axline((0, 0), slope=1, color="gray", linestyle="--", linewidth=1)
    axes[0, 2].axline((0, 0), slope=1, color="gray", linestyle="--", linewidth=1)

    axes[1, 0].axline((0,0), slope=0, color="gray", linestyle="--", linewidth=1)
    axes[1, 1].axline((0,0), slope=0, color="gray", linestyle="--", linewidth=1)
    axes[1, 2].axline((0,0), slope=0, color="gray", linestyle="--", linewidth=1)

    axes[0, 0].set_xlabel(rf"True $kT$ [keV]")
    axes[0, 0].set_ylabel(r"Predicted $kT$ [keV]")

    axes[0, 1].set_xscale("log")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xlabel(r"True flux [$\mathrm{erg}\,\mathrm{cm}^{-2}\,\mathrm{s}^{-1}$]")
    axes[0, 1].set_ylabel(r"Predicted flux [$\mathrm{erg}\,\mathrm{cm}^{-2}\,\mathrm{s}^{-1}$")

    axes[0, 2].set_xscale("log")
    axes[0, 2].set_yscale("log")
    axes[0, 2].set_xlabel(r"True $N_\mathrm{H}$ [$10^{22}\,\mathrm{cm}^{-2}$]")
    axes[0, 2].set_ylabel(r"Predicted $N_\text{H}$ [$10^{22}\,\mathrm{cm}^{-2}$]")

    axes[1, 0].set_xscale("log")
    axes[1, 0].set_xlabel(r"True flux [$\mathrm{erg}\,\mathrm{cm}^{-2}\,\mathrm{s}^{-1}$]")
    axes[1, 0].set_ylabel(r"$(kT_\mathrm{pred.}-kT_\mathrm{true})/kT_\mathrm{true}$")

    axes[1, 1].set_xscale("log")
    axes[1, 1].set_xlabel(r"True flux [$\mathrm{erg}\,\mathrm{cm}^{-2}\,\mathrm{s}^{-1}$]")
    axes[1, 1].set_ylabel(r"$(\mathrm{Flux}_\mathrm{pred.}-\mathrm{Flux}_\mathrm{true})/\mathrm{Flux}_\mathrm{true}$")

    axes[1, 2].set_xscale("log")
    axes[1, 2].set_xlabel(r"True flux [$\mathrm{erg}\,\mathrm{cm}^{-2}\,\mathrm{s}^{-1}$]")
    axes[1, 2].set_ylabel(r"$(N_\text{H, pred.}-N_\text{H, true})/N_\text{H, true}$")

    axes[0, 0].legend()

    plt.tight_layout()
    plt.savefig("testdata.pdf")

def plot_2d_posteriors(model, dataset, scaler = None, num_samples=10000, device='cpu'):
    model.eval()
    indices = [int(random.uniform(0, len(dataset) - 1)) for _ in range(20)]
    outfiles = []
    names = [LABEL_DICT["kt"], LABEL_DICT["src_flux"], LABEL_DICT["nh"]]

    for idx in indices:
        x, y_true = dataset[idx]
        x = x.unsqueeze(0).to(device)  # add batch‐dim

        #input_fname, target_fname = dataset.get_filenames(idx)

        with torch.no_grad():
            q_dist = model(x)
            samples = q_dist.sample((num_samples,))

        samples = samples.squeeze(1).cpu().numpy()  # remove batch dimension

        if scaler:
            samples = scaler.inverse_transform(samples)
            y_true = scaler.inverse_transform(y_true.reshape(1, -1)).flatten()

            # Convert log(flux) back to linear flux
            samples[:, 1] = 10**samples[:, 1]
            y_true[1] = 10**y_true[1]

        figure = corner.corner(samples, labels=names, smooth=1,
                               show_titles=False, title_kwargs={"fontsize": 12},
                               bins=100, truths=y_true, range=[0.999, 0.999, 0.999])
        #figure.suptitle(os.path.basename(input_fname))

        outfile = f"outfiles/testdata_{idx}.pdf"
        outfiles.append(outfile)
        plt.tight_layout()
        plt.savefig(outfile)

    unite_pdfs(outfiles, outfilename="posteriors.pdf")


def main():
    def plot_testdata():
        train_dataset, val_dataset, test_dataset = load_and_split_dataset()

        model = ConvSpectraFlow()
        model.load_state_dict(torch.load(ml_config.data_neural_network + "model_weights.pth", weights_only=True, map_location="cpu"))
        scaler = joblib.load(ml_config.data_neural_network + "target_scaler.pkl")

        evaluate_on_test_spectrum(model, test_dataset, plot_input_data_only=True)
        # evaluate_on_real_spectrum(model, "/pool/burg1/novae4ole/V1710Sco_em04_PATall_820_SourceSpec_00001.fits", out_pha_file = "test.fits")
        # evaluate_on_real_spectrum(model, "/pool/burg1/tmp/YZRet_Nova_Fireball_020_SourceSpec_00001.fits")
        # evaluate_parameter_prediction(model, test_dataset)
        #plot_2d_posteriors(model, test_dataset, scaler = scaler)

    def plot_loss_from_csv():
        metadata = pd.read_csv("../neural_network/loss.csv")
        plot_loss(metadata, label=r"Loss $[-\log q(\theta | x)]$")

    # plot_loss_from_csv()
    plot_testdata()

if __name__ == "__main__":
    main()
