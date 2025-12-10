import csv
import numpy as np
from dataclasses import dataclass,fields
import torch
from torch.utils.data import DataLoader
import joblib

from code.config import MLConfig
from code.neural_network.data import load_and_split_dataset
from code.subs import plot_loss
from code.neural_network.neuralnetwork import ConvSpectraFlow

device = 'cpu'
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

def training_loop(model, train_loader, criterion, optimizer):
    model.train()  # Set the model to training mode
    running_train_loss = 0.0
    batch_losses = []
    total_grad_norms = []

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # Move data to the correct device (GPU/CPU)
        inputs, targets = inputs.to(device), targets.to(device)

        loss = model.nll(inputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()  # Zero the gradients

        try:
            loss.backward()
        except Exception as e:
            print(f"Error {e} at batch {batch_idx}, loss={loss.item()}")
            raise

        total_norm = _get_grad_norm(model)
        total_grad_norms.append(total_norm)

        optimizer.step()

        batch_losses.append(loss.item())  # Store batch-wise loss
        running_train_loss += loss.item() * inputs.size(0)

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

            loss = model.nll(inputs, targets)

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

    train_loader = DataLoader(train_dataset, batch_size=ml_config.batch_size, shuffle=True, num_workers=ml_config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=ml_config.batch_size, shuffle=False, num_workers=ml_config.num_workers)

    model = ConvSpectraFlow()
    model.to(device)

    #model.load_state_dict(torch.load(ml_config.data_neural_network + "model_weights.pth", map_location="cpu"))

    criterion = False  # use for ConvSpectraFlow

    optimizer = torch.optim.Adam(model.parameters(), lr=ml_config.learning_rate, weight_decay=0)
    metadata = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=ml_config.num_epochs)

    joblib.dump(train_dataset.scaler, ml_config.data_neural_network + "target_scaler.pkl")

    model_file = ml_config.data_neural_network + "model_weights.pth"
    torch.save(model.state_dict(), model_file)
    print(f"Trained model saved to {model_file}")

    write_metadata_file(metadata)
    plot_loss(metadata, label = type(criterion).__name__)



if __name__ == "__main__":
    main()
