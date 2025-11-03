import csv
import numpy as np
from dataclasses import dataclass,fields

import torch
from torch.utils.data import DataLoader
import joblib

from data import load_and_split_dataset
from config import MLConfig
from subs import plot_loss

from neuralnetwork import ConvSpectraFlow

device = 'cpu'  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ml_config = MLConfig()

@dataclass
class EpochMetadata:
    running_train_loss: float
    mean_total_grad_norm: float
    batch_losses: list[float]
    total_grad_norms: list[float]

@dataclass
class TrainMetadata:
    training_losses: list[float]
    validation_losses: list[float]
    running_train_losses: list[float]
    mean_total_grad_norms: list[float]
    learning_rate: list[float]

# FC4_PRE_ACTIVATION = None
# def _capture_fc4_pre_activation(module, input, output):
#     """Define a hook function to capture the output of fc4 before the activation is applied."""
#     global FC4_PRE_ACTIVATION
#     # Capture a clone of the output to avoid any in-place issues.
#     FC4_PRE_ACTIVATION = output.detach().clone()

def _get_grad_norm(model: torch.nn.Module) -> float:
    grad_norms = []
    total_norm = 0
    for param in model.parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm(2).item()  # L2 norm of the gradients
            grad_norms.append(grad_norm)
            total_norm += grad_norm**2
    total_norm = total_norm**0.5
    return total_norm

#def _get_mu_var_from_model(outputs):
#    """Split parameter means (first numbers) and log variance (last half of numbers).
#    It does not quite correspond to the variance because I'm applying softplus
#    instead of exp on the logarithm of the variance for numerical stability."""
#    mu, raw_log_var = outputs[:, :ml_config.dim_output_parameters], outputs[:, ml_config.dim_output_parameters:]
#    floor = -10
#    raw_log_var = raw_log_var.clamp(floor, 6)  # Clamp such that variance doesn't explode to zero/infinity
#    var = torch.nn.functional.softplus(raw_log_var)
#    if var.min() < np.exp(floor+1):
#        print(f"WARNING: Log variance seems to hit the clamped floor")
#    return mu, var

#def _transform_targets(targets: torch.Tensor) -> torch.Tensor:
#    """
#    Prediction of log‑values to avoid negative values + capture large value range
#    """
#    epsilon = 1e-6
#    mu0 = torch.log(targets[:, 0].clamp_min(epsilon))
#    mu1 = torch.log(targets[:, 1].clamp_min(epsilon))
#    mu2 = torch.log(targets[:, 2].clamp_min(epsilon))
#    log_targets = torch.stack([mu0, mu1, mu2], dim=1)
#    return log_targets

def training_loop(model, train_loader, criterion, optimizer):
    model.train()  # Set the model to training mode
    running_train_loss = 0.0
    batch_losses = []
    total_grad_norms = []

    #hook_fc4 = model.fc4.register_forward_hook(_capture_fc4_pre_activation)
    #lambda_reg = 1e-10  # regularization coefficient to penalize negative pre-activations.

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # Move data to the correct device (GPU/CPU)
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        # outputs = model(inputs)

        # loss = criterion(outputs, log_targets)

        # mu, var = _get_mu_var_from_model(outputs)
        # log_targets = _transform_targets(targets)
        # loss = criterion(mu, log_targets, var)  # use for GaussianNLLLoss
        # loss = criterion(outputs, targets)

        loss = model.nll(inputs, targets)

        # Add custom regularization on last layer's pre-activation outputs
        # (to penalize negative values at high energies):
        #if FC4_PRE_ACTIVATION is not None:
        #    # We want to penalize any negative values: clamp the negative part and square it.
        #    reg_term = lambda_reg * torch.sum(torch.clamp(-1 * FC4_PRE_ACTIVATION, min=0) ** 2)
        #    print(loss.item(), reg_term.item())
        #    loss = loss + reg_term

        # Backward pass and optimization
        optimizer.zero_grad()  # Zero the gradients

        try:
            loss.backward()
        except Exception as e:
            print(f"Error {e} at batch {batch_idx}, loss={loss.item()}")
            raise

        total_norm = _get_grad_norm(model)
        total_grad_norms.append(total_norm)

        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        batch_losses.append(loss.item())  # Store batch-wise loss
        running_train_loss += loss.item() * inputs.size(0)

    #hook_fc4.remove()

    running_train_loss = running_train_loss / len(train_loader.dataset)
    mean_total_grad_norm = np.mean(total_grad_norms)

    metadata = EpochMetadata(running_train_loss = running_train_loss,
                             mean_total_grad_norm = mean_total_grad_norm,
                             batch_losses = batch_losses,
                             total_grad_norms = total_grad_norms)

    return metadata

def validation_loop(model, val_loader, criterion):
    model.eval()  # Set the model to evaluation mode
    running_val_loss = 0.0

    with torch.no_grad():  # Disable gradient computation during evaluation
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            # loss = criterion(outputs, targets)  # use for MSELoss / PoissonNLLLoss

            # mu, var = _get_mu_var_from_model(outputs)
            #log_targets = _transform_targets(targets)
            # loss = criterion(mu, log_targets, var)  # use for GaussianNLLLoss

            loss = model.nll(inputs, targets)  # use for ConvSpectraFlow

            # ################################################
            # Manual calculation of loss
            # term1 = 0.5 * torch.log(var)
            # term2 = 0.5 * ((mu - log_targets) ** 2 / var)
            # mean_term1 = term1.mean().detach()
            # mean_term2 = term2.mean().detach()
            # term1s += mean_term1.item() * inputs.size(0)
            # term2s += mean_term2.item() * inputs.size(0)
            # ################################################

            running_val_loss += loss.item() * inputs.size(0)  # Scale by batch size

    avg_val_loss = running_val_loss / len(val_loader.dataset)  # Normalize by total samples

    return avg_val_loss

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=8):
    train_metadata = TrainMetadata(*[list() for dummy in fields(TrainMetadata)])

    for epoch in range(num_epochs):
        epoch_metadata = training_loop(model, train_loader, criterion, optimizer)

        train_metadata.running_train_losses.append(epoch_metadata.running_train_loss)
        train_metadata.mean_total_grad_norms.append(epoch_metadata.mean_total_grad_norm)

        avg_val_loss = validation_loop(model, val_loader, criterion)

        train_metadata.validation_losses.append(avg_val_loss)
        avg_train_loss = validation_loop(model, train_loader, criterion)
        train_metadata.training_losses.append(avg_train_loss)

        train_metadata.learning_rate.append(ml_config.learning_rate)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

    return train_metadata

def write_metadata_file(metadata, csv_filename = "loss.csv"):
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        header = ["Epoch"] + [field.name for field in fields(metadata)]
        writer.writerow(header)
        all_fields = [getattr(metadata, field.name) for field in fields(metadata)]
        for epoch, values in enumerate(zip(*all_fields), start=1):
            writer.writerow([epoch] + list(values))

def main():
    train_dataset, val_dataset, test_dataset = load_and_split_dataset()
    print(f"Number of training samples: {len(train_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=ml_config.batch_size, shuffle=True, num_workers=32)
    val_loader = DataLoader(val_dataset, batch_size=ml_config.batch_size, shuffle=False, num_workers=32)

    model = ConvSpectraFlow()
    model.to(device)

    #model.load_state_dict(torch.load(ml_config.data_neural_network + "model_weights.pth", map_location="cpu"))

    # criterion = torch.nn.MSELoss()  # use for parameter estimator
    # criterion = torch.nn.PoissonNLLLoss(log_input=False, full=True, reduction='mean')  # use for spectral estimator
    # criterion = torch.nn.GaussianNLLLoss(eps=1e-6, reduction='mean')  # use for parameter + variance estimator
    criterion = False  # use for ConvSpectraFlow

    optimizer = torch.optim.Adam(model.parameters(), lr=ml_config.learning_rate, weight_decay=0)
    metadata = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=2048)

    joblib.dump(train_dataset.scaler, ml_config.data_neural_network + "target_scaler.pkl")

    model_file = ml_config.data_neural_network + "model_weights.pth"
    torch.save(model.state_dict(), model_file)
    print(f"Trained model saved to {model_file}")

    write_metadata_file(metadata)
    plot_loss(metadata, label = type(criterion).__name__)



if __name__ == "__main__":
    main()