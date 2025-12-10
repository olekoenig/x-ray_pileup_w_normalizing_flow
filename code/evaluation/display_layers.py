import torch
import matplotlib.pyplot as plt
import os

from code.neural_network.data import load_and_split_dataset
from code.neural_network.neuralnetwork import pileupNN
import code.config


ACTIVATIONS = {}

def get_activation_hook(layer_name):
    def hook_fn(module, input, output):
        ACTIVATIONS[layer_name] = output.detach()  # Use detach() to avoid unnecessary computation graph retention
    return hook_fn

def plot_last_layer(model):
    """Function used to manually check the matrix multiplication of weights and feature vector in the last layer."""
    train_dataset, val_dataset, test_dataset = load_and_split_dataset()
    input_data, target_data = test_dataset[0]

    global ACTIVATIONS
    ACTIVATIONS = {}  # clear

    hook_fc3 = model.fc3.register_forward_hook(get_activation_hook('penultimate_layer'))
    hook_fc4 = model.fc4.register_forward_hook(get_activation_hook('last_layer'))

    model.eval()
    with torch.no_grad():
        predicted_output = model(input_data)

    penultimate_data = ACTIVATIONS['penultimate_layer']
    last_layer_data = ACTIVATIONS['last_layer']

    fig, axes = plt.subplots(sharex=True, ncols=1, nrows=2, figsize=(15 / 2.54, 10 / 2.54))

    # Apply activation to the last layer (hook returns the values *before* the activation function)
    softplus = torch.nn.Softplus()
    penultimate_activated = softplus(penultimate_data.cpu())

    fc4_wo_bias = torch.matmul(penultimate_activated, model.fc4.weight.cpu().t())
    bias_fc4 = model.fc4.bias.data.cpu()
    manual_fc4 = fc4_wo_bias + bias_fc4
    manual_fc4_activated = softplus(manual_fc4)

    axes[0].plot(fc4_wo_bias.detach().numpy(), label=r'$x_\mathrm{fc3} * w^T$', color="cyan", linewidth=2)
    axes[0].plot(manual_fc4.detach().numpy(), label=r'$x_\mathrm{fc3} * w^T + b$', color = "blue", linewidth=1)
    axes[0].plot(manual_fc4_activated.detach().numpy(), label=r'$\sigma (x_\mathrm{fc3} * w^T + b)$', color="red", linewidth=1)

    axes[0].plot(last_layer_data.detach().numpy(), label=r'$x_\mathrm{fc4}$', color='navy', linewidth=1)
    axes[0].plot(predicted_output, label=r'Predicted output', color='maroon', linewidth=1)

    axes[1].plot(bias_fc4, label=r'$b$', color='black', linewidth=1)
    axes[-1].set_xlabel('Neuron Index')

    axes[0].set_xscale('log')
    axes[1].set_xscale('log')

    axes[0].legend()
    axes[1].legend()
    plt.tight_layout()
    plt.savefig("last_layer.pdf")

    hook_fc3.remove()
    hook_fc4.remove()

def plot_activations(model):
    train_dataset, val_dataset, test_dataset = load_and_split_dataset()
    index = 20
    input_data, target_data = test_dataset[index]
    input_fname, target_fname = test_dataset.get_filenames(index)

    global ACTIVATIONS
    ACTIVATIONS = {}  # clear

    hook_fc1 = model.fc1.register_forward_hook(get_activation_hook('fc1'))
    hook_fc2 = model.fc2.register_forward_hook(get_activation_hook('fc2'))
    hook_fc3 = model.fc3.register_forward_hook(get_activation_hook('penultimate_layer'))
    hook_fc4 = model.fc4.register_forward_hook(get_activation_hook('last_layer'))

    # Run the evaluation step (forward pass)
    model.eval()
    with torch.no_grad():
        predicted_output = model(input_data)

    # Extract the activations of the layers (after the forward pass)
    fc1_data = ACTIVATIONS['fc1']
    fc2_data = ACTIVATIONS['fc2']
    penultimate_data = ACTIVATIONS['penultimate_layer']
    last_data = ACTIVATIONS['last_layer']

    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(15/2.54, 20/2.54))
    fig.suptitle(os.path.basename(input_fname))

    axes[0].plot(input_data, color='black')
    axes[1].plot(fc1_data.cpu().numpy(), color='orange')
    axes[2].plot(fc2_data.cpu().numpy(), color='green')
    axes[3].plot(penultimate_data.cpu().numpy(), color='magenta')

    axes[-1].plot([0, len(last_data.cpu().numpy())], [0, 0], linestyle='--', color='gray')
    activation_name = type(model.activation).__name__
    axes[-1].plot(last_data.cpu().numpy(), label=f'Activation before {activation_name}', color='blue', linewidth=2)
    axes[-1].plot(predicted_output, label=f'Predicted output (after {activation_name})', color='red', linewidth=1)
    axes[-1].legend()

    axes[0].set_ylabel('Input data')
    axes[1].set_ylabel('First layer (fc1)')
    axes[2].set_ylabel('Second layer (fc2)')
    axes[3].set_ylabel('Penultimate layer (fc3)')
    axes[-1].set_ylabel('Last layer (fc4)')
    axes[-1].set_xlabel('Neuron Index')

    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[-1].set_xscale('log')

    outfile = "activations.pdf"
    plt.savefig(outfile)
    print(f"Wrote {outfile}")

    hook_fc1.remove()
    hook_fc2.remove()
    hook_fc3.remove()
    hook_fc4.remove()

def plot_weights_and_biases(model):
    weights_fc4 = model.fc4.weight.data.cpu()
    weights_fc4_transpose = weights_fc4.t().numpy()
    bias_fc4 = model.fc4.bias.data.cpu().numpy()

    nx = 4
    ny = int(len(weights_fc4_transpose)/4)
    fig, axes = plt.subplots(ncols=nx, nrows=ny+1, sharex=False, figsize=(nx*4, ny*2.2))
    fig.suptitle('Weights and biases of last layer (fc4)')
    fig.supxlabel('Neuron Index')
    fig.supylabel('Value')

    neuron_idx = 0
    for x in range(nx):
        for y in range(ny):
            axes[y, x].plot(weights_fc4_transpose[neuron_idx], linestyle='-', label=f'Weights (Neuron {neuron_idx})')
            axes[y, x].legend(loc='upper right')
            axes[y, x].set_xscale('log')
            neuron_idx += 1

    axes[ny, 0].plot(bias_fc4, color="black", label="Bias")
    axes[ny, 0].legend(loc='upper right')
    axes[ny, 0].set_xscale('log')
    [axes[ny, ii].set_axis_off() for ii in range(1, nx)]

    plt.tight_layout()
    outname = "weights_and_biases.pdf"
    plt.savefig(outname)
    print(f"Wrote {outname}")

def main():
    model = pileupNN()
    model.load_state_dict(torch.load(config.DATA_NEURAL_NETWORK + "model_weights.pth", map_location="cpu"))

    plot_activations(model)
    # plot_last_layer(model)
    # plot_weights_and_biases(model)


if __name__ == '__main__':
    main()
