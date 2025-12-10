import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
#import corner
import joblib
import os
import random

from data import load_and_split_dataset
from neuralnetwork import ConvSpectraFlow
from config import MLConfig, ModelConfig
from subs import load_emcee_fits, unite_pdfs

ml_config = MLConfig()
model_config = ModelConfig()

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'


class NFPosteriorEvaluator:
    def __init__(self, model, dataset, scaler = None, num_samples=10000):
        self.model = model
        self.dataset = dataset
        self.num_samples = num_samples
        self.results = []
        self.scaler = scaler

    def get_samples(self, dataset):
        x, y_true = dataset
        x = x.unsqueeze(0)

        y_true = y_true.cpu().numpy()

        with torch.no_grad():
            q_dist = self.model(x)
            samples = q_dist.sample((self.num_samples,))

        samples = samples.squeeze(1).cpu().numpy()

        if self.scaler:
            samples = self.scaler.inverse_transform(samples)
            y_true = self.scaler.inverse_transform(y_true.reshape(1, -1)).flatten()
            samples[:, 1] = 10 ** samples[:, 1]
            y_true[1] = 10 ** y_true[1]

        return samples, y_true

    def evaluate_posterior_accuracy(self):
        self.model.eval()
        print(f"Evaluating {len(self.dataset)} test samples...")

        for idx in tqdm(range(len(self.dataset))):
            samples, y_true = self.get_samples(self.dataset[idx])
            sample_results = self._evaluate_single_posterior(samples, y_true, idx)
            self.results.append(sample_results)

        return self._compile_results()

    def _evaluate_single_posterior(self, samples, y_true, idx):
        sample_results = {
            'idx': idx,
            'ground_truth': y_true.copy(),
            'posterior_stats': {}
        }

        for param_idx in range(ml_config.dim_output_parameters):
            param_samples = samples[:, param_idx]
            true_value = y_true[param_idx]

            # param_samples <= true_value creates a boolean array (True/False for each sample)
            # np.mean() treats True as 1, False as 0, and computes the average
            # -> gives the fraction of posterior samples that are smaller/equal to the true value
            cdf_value = np.mean(param_samples <= true_value)

            mean = np.mean(param_samples)
            median = np.median(param_samples)
            std = np.std(param_samples)
            bias = median - true_value
            relative_bias = bias / true_value
            z_score = (true_value - mean) / std

            param_results = {
                'mean': mean,
                'median': median,
                'std': std,
                'true_value': true_value,
                'bias': bias,
                'relative_bias': relative_bias,
                'z_score': z_score,
                'cdf_value': cdf_value,
            }

            sample_results['posterior_stats'][f'param_{param_idx}'] = param_results

        return sample_results

    def _compile_results(self):
        compiled = {'per_parameter_stats': {}}

        for param_idx in range(ml_config.dim_output_parameters):
            cdf_values = []
            relative_biases = []
            z_scores = []
            true_values = []

            for result in self.results:
                param_data = result['posterior_stats'][f'param_{param_idx}']
                cdf_values.append(param_data['cdf_value'])
                relative_biases.append(param_data['relative_bias'])
                z_scores.append(param_data['z_score'])
                true_values.append(param_data['true_value'])

            compiled['per_parameter_stats'][f'param_{param_idx}'] = {
                'cdf_values': cdf_values,
                'mean_relative_bias': np.mean(np.abs(relative_biases)),
                'rms_z_score': np.sqrt(np.mean(np.array(z_scores) ** 2)),
                'relative_biases': relative_biases,
                'z_scores': z_scores,
                'true_values': true_values
            }

        return compiled

    def plot_cdf_histograms(self, results):
        fig, axes = plt.subplots(1, ml_config.dim_output_parameters, figsize=(10, 10/3))

        for param_idx in range(ml_config.dim_output_parameters):
            cdf_values = results['per_parameter_stats'][f'param_{param_idx}']['cdf_values']

            axes[param_idx].hist(cdf_values, bins=20, alpha=0.7, density=False,
                                 color=model_config.colors[param_idx], edgecolor='black')
            axes[param_idx].set_xlabel('Percentiles (mean(samples $\leq$ ground truth)) for ' + model_config.names[param_idx])

            axes[param_idx].set_ylabel('Number of samples')

        plt.tight_layout()
        plt.savefig("cdf_histograms.pdf")
        print("Wrote cdf_histograms.pdf")

    def plot_relative_bias(self, results):
        flux_true = results['per_parameter_stats']['param_1']['true_values']
        fig, axes = plt.subplots(1, ml_config.dim_output_parameters, figsize=(12, 3))

        for param_idx in range(ml_config.dim_output_parameters):
            relative_bias = results['per_parameter_stats'][f'param_{param_idx}']['relative_biases']

            axes[param_idx].scatter(flux_true, relative_bias, alpha=0.1, s=10, color=model_config.colors[param_idx])
            axes[param_idx].set_xscale('log')
            axes[param_idx].set_xlabel(f'True {model_config.names[1]} [{model_config.units[1]}]')
            axes[param_idx].set_ylabel(r'Relative error: ' + model_config.names[param_idx])

            # cut out some percentage of points for plotting due to very few outliers (only for visualization)
            perc = 0.02
            ranges = np.percentile(relative_bias, [perc, 100-perc])
            axes[param_idx].set_ylim(ranges[0], ranges[1])

            #cdf_values = results['per_parameter_stats'][f'param_{param_idx}']['cdf_values']
            #axes[1, param_idx].scatter(flux_true, cdf_values, alpha=0.5, s=10, color=model_config.colors[param_idx])
            #axes[1, param_idx].set_xscale('log')
            #axes[1, param_idx].set_xlabel(f'True {model_config.names[1]} [{model_config.units[1]}]')
            #axes[1, param_idx].set_ylabel(r'Percentiles (mean(samples $\leq$ ground truth)) for' + model_config.names[param_idx])

            # https://en.wikipedia.org/wiki/Mean_absolute_percentage_error
            mape = results['per_parameter_stats'][f'param_{param_idx}']['mean_relative_bias'] * 100
            axes[param_idx].text(0.5, 0.95, f'Mean absolute percentage error: {mape:.1f}\%',
                                 transform=axes[param_idx].transAxes,
                                 horizontalalignment='center', verticalalignment='top',
                                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.subplots_adjust(hspace=0)
        plt.tight_layout()
        plt.savefig("relative_bias.pdf")
        print("Wrote relative_bias.pdf")

    def plot_sbc_rank_comparison(self, results):
        from sbi.analysis.plot import sbc_rank_plot
        num_tests = len(self.results)
        ranks = np.zeros((num_tests, 3), dtype=int)

        for param_idx in range(3):
            pit_values = np.array(results['per_parameter_stats'][f'param_{param_idx}']['cdf_values'])
            # Convert PIT values to ranks
            ranks[:, param_idx] = (pit_values * self.num_samples).astype(int)

        fig, axes = sbc_rank_plot(
            ranks=ranks,
            num_posterior_samples=self.num_samples,
            plot_type="cdf",
            parameter_labels=model_config.names
        )

        return fig, axes

    def plot_coverage(self, results):
        fig, axes = plt.subplots(1, 3, figsize=(12, 12/3))

        coverage_levels = np.arange(0.0, 1.1, 0.1)

        kt_true = np.array(results['per_parameter_stats']['param_0']['true_values'])
        flux_true = np.array(results['per_parameter_stats']['param_1']['true_values'])
        nh_true = np.array(results['per_parameter_stats']['param_2']['true_values'])

        lookup = {
            'trues': [kt_true, flux_true, nh_true],
            'low': [[0.03, 0.05, 0.1], [1e-12, 1e-11, 1e-10, 1e-9], [0.2, 1]],
            'high': [[0.05, 0.1, 0.2], [1e-11, 1e-10, 1e-9, 1e-8], [1, 2]],
        }
        symbols = ["v", "s", "D", "*"]

        for param_idx in range(ml_config.dim_output_parameters):
            # Get percentiles (from CDF) for this parameter
            cdf_values = np.array(results['per_parameter_stats'][f'param_{param_idx}']['cdf_values'])

            axes[param_idx].plot([0, 1], [0, 1], '--', color="gray", linewidth=1,
                                 label='Perfect calibration')

            color = [0, 0, 0]  # (r,g,b)

            # Plot coverage in this parameter range
            nbins = len(lookup['low'][param_idx])
            for bin_idx in range(nbins):
                true = lookup['trues'][param_idx]
                lo = lookup['low'][param_idx][bin_idx]
                hi = lookup['high'][param_idx][bin_idx]

                # Mask out this parameter range (e.g., low fluxes)
                mask = ((true >= lo) & (true < hi))
                cdf_values_in_bin = cdf_values[mask]

                empirical_coverage_bin = []
                for coverage_level in coverage_levels:
                    # Calculate symmetric interval bounds
                    lower_bound = (1 - coverage_level) / 2
                    upper_bound = (1 + coverage_level) / 2
                    in_interval = (cdf_values_in_bin >= lower_bound) & (cdf_values_in_bin <= upper_bound)
                    empirical_cov = np.mean(in_interval)

                    # "Integrate from left (PIT-approach from Eq. 7 in Dalmosso+20)
                    #cumulative_coverage_bin = np.mean(cdf_values_in_bin <= coverage_level)

                    empirical_coverage_bin.append(empirical_cov)

                ece_bin = np.mean(np.abs(empirical_coverage_bin - coverage_levels))

                if param_idx == 0:
                    label = fr'${lo}$--${hi}$\,{model_config.units[param_idx]}'
                if param_idx == 1:
                    #label = fr'$10^{{{int(np.log10(lo))}}}$--$10^{{{int(np.log10(hi))}}}$\,{model_config.units[param_idx]}'
                    label = fr'$10^{{{int(np.log10(lo))}}}$--$10^{{{int(np.log10(hi))}}}\,\mathrm{{cgs}}$'
                if param_idx == 2:
                    label = fr'$({lo}$--${hi})\,\times$\,{model_config.units[param_idx]}'

                color[param_idx] = 0.8 - bin_idx * 0.6/nbins
                axes[param_idx].plot(coverage_levels, empirical_coverage_bin, '-',
                                     marker = symbols[bin_idx], color=tuple(color),
                                     markersize=4, linewidth=2, alpha = 0.7,
                                     label=label)

                #axes[param_idx].text(0.65, 0.27-bin_idx*0.06, f'ECE = {ece_bin:.3f}%',
                #                     transform=axes[param_idx].transAxes,
                #                     color=tuple(color))


            # Now plot over whole dataset (not range-resolved)
            empirical_coverage = []
            for coverage_level in coverage_levels:
                # "Integrating from left" (PIT approach)
                #cumulative_coverage = np.mean(cdf_values <= coverage_level)

                # Integrate symmetrically outwards from center of distribution
                lower_bound = (1 - coverage_level) / 2
                upper_bound = (1 + coverage_level) / 2
                in_interval = (cdf_values >= lower_bound) & (cdf_values <= upper_bound)
                empirical_cov = np.mean(in_interval)
                empirical_coverage.append(empirical_cov)

            ece = np.mean(np.abs(empirical_coverage - coverage_levels))

            axes[param_idx].plot(coverage_levels, empirical_coverage, 'o-', color=model_config.colors[param_idx],
                                 markersize=6, linewidth=3, label=f'Total test dataset')
            #axes[param_idx].text(0.65, 0.27-nbins * 0.06, fr'ECE$_\mathrm{{total}}$ = {ece:.3f}',
            #                     transform=axes[param_idx].transAxes,
            #                     color=tuple(color))

            axes[param_idx].set_xlabel('Confidence level')
            axes[param_idx].set_ylabel('Empirical coverage')
            axes[param_idx].set_title(f'Coverage plot: {model_config.names[param_idx]}')
            axes[param_idx].legend(loc='upper left')
            axes[param_idx].set_xlim(0, 1)
            axes[param_idx].set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig("coverage.pdf")
        print("Wrote coverage.pdf")

        return fig

    def print_summary(self, results):
        print("\n" + "=" * 60)
        print("NORMALIZING FLOW CALIBRATION SUMMARY")
        print("=" * 60)

        param_names = model_config.names
        for param_idx in range(ml_config.dim_output_parameters):
            param_stats = results['per_parameter_stats'][f'param_{param_idx}']

            #cdf_values = param_stats['cdf_values']
            #ks_stat, ks_p = stats.kstest(cdf_values, 'uniform')

            print(f"{param_names[param_idx]}:")
            #print(f"    p uniformity (KS test): stat={ks_stat:.3f}, p={ks_p:.3f}")
            print(f"    Mean relative bias: {param_stats['mean_relative_bias']:.4f}")
            print(f"    RMS Z-score: {param_stats['rms_z_score']:.3f}")

    def plot_2d_posteriors_with_mcmc(self, input_name, mcmc_file, ranges, color, label):
        ranges = ranges
        color = color
        label = label

        nbins = 100
        names = [rf"{model_config.names[0]}\,[{model_config.units[0]}]",
                 rf"{model_config.names[2]}\,[{model_config.units[2]}]"]

        self.model.eval()
        idx = self.dataset.index_of_input(input_name)
        samples, y_true = self.get_samples(self.dataset[idx])

        mcmc = load_emcee_fits(mcmc_file, burn_in=1000)
        figure = corner.corner(mcmc,
                               labels=names, bins=nbins,
                               range=ranges,
                               color=color,
                               plot_datapoints=True,
                               hist_kwargs={'density': True, 'color': color},
                               data_kwargs={'alpha': 0.01})

        # Get kT and NH only
        samples = samples[:, [0, 2]]
        y_true = y_true[[0, 2]]

        corner.corner(samples,
                      fig=figure,
                      labels=names,
                      color="black",
                      show_titles=False,
                      bins=50,
                      # smooth=1,
                      truths=y_true, truth_color='blue',
                      range=ranges,
                      hist_kwargs={'density': True, 'color': 'black'},
                      data_kwargs={'alpha': 0.5})

        basename = os.path.basename(input_name)
        figure.suptitle(basename)

        handles = [Patch(color=color, label=label),
                   Patch(color='black', label="NF posterior"),
                   Patch(color='blue', label="Ground truth")]

        axes = np.array(figure.axes).reshape((2, 2))
        axes[1, 0].legend(handles=handles, loc='upper right')

        plt.tight_layout()
        outfile = "cmp_flow_to_mcmc_" + basename.split("_circle0.fits")[0] + ".pdf"
        plt.savefig(outfile)
        print(f"Wrote {outfile}")

    def plot_2d_posteriors(self):
        self.model.eval()
        indices = [int(random.uniform(0, len(self.dataset) - 1)) for _ in range(20)]
        outfiles = []
        names = [rf"{model_config.names[param_idx]}\,[{model_config.units[param_idx]}]" for param_idx in range(ml_config.dim_output_parameters)]

        for idx in indices:
            samples, y_true = self.get_samples(self.dataset[idx])

            fig = corner.corner(samples, labels=names, #smooth=1,
                                # show_titles=False, title_kwargs={"fontsize": 12},
                                bins=100, truths=y_true, range=[0.999, 0.999, 0.999])

            input_fname = self.dataset[idx]
            print(f"\tPlotting {input_fname}")
            fig.suptitle(os.path.basename(input_fname))

            outfile = f"testdata_{idx}.pdf"
            outfiles.append(outfile)
            plt.tight_layout()
            plt.savefig(outfile)

        unite_pdfs(outfiles, outfilename="posteriors.pdf")


def main():
    train_dataset, val_dataset, test_dataset = load_and_split_dataset()
    model = ConvSpectraFlow()
    model.load_state_dict(torch.load(ml_config.data_neural_network + "model_weights.pth",
                                     weights_only=True, map_location="cpu"))
    scaler = joblib.load(ml_config.data_neural_network + "target_scaler.pkl")

    #from torch.utils.data import Subset
    #test_dataset = Subset(test_dataset, range(10))

    evaluator = NFPosteriorEvaluator(model, test_dataset, scaler=scaler, num_samples=10000)

    results = evaluator.evaluate_posterior_accuracy()
    evaluator.print_summary(results)

    #evaluator.plot_cdf_histograms(results)
    #evaluator.plot_relative_bias(results)
    evaluator.plot_coverage(results)

if __name__ == "__main__":
    main()
