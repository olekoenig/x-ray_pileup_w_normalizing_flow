import torch
import joblib

from data import load_and_split_dataset
from neuralnetwork import ConvSpectraFlow
from nfposteriorevaluator import NFPosteriorEvaluator
from config import MLConfig

ml_config = MLConfig()

def main():
    train_dataset, val_dataset, test_dataset = load_and_split_dataset()
    model = ConvSpectraFlow()
    model.load_state_dict(torch.load(ml_config.data_neural_network + "model_weights.pth",
                                     weights_only=True, map_location="cpu"))
    scaler = joblib.load(ml_config.data_neural_network + "target_scaler.pkl")

    evaluator = NFPosteriorEvaluator(model, test_dataset, scaler=scaler, num_samples=10000)
    results = evaluator.evaluate_posterior_accuracy()

    evaluator.plot_coverage(results)
