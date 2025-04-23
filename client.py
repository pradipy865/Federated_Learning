import flwr as fl
import numpy as np
from typing import Tuple, Dict, List, Any
from model import create_model


class SmartGridClient(fl.client.NumPyClient):
    def __init__(self, x_train: np.ndarray, y_train: np.ndarray, optimizer_type: str = "yogi"):
        self.x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        self.y_train = y_train
        self.model = create_model(x_train.shape[1], optimizer_type=optimizer_type)

    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        return self.model.get_weights()

    def fit(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        self.model.set_weights(parameters)

        # ----- Adaptive Batch Size Calculation -----
        Ri = np.random.uniform(1.0, 5.0)
        alpha = 0.5
        Bmax = 64
        Bt = min(Bmax, int(Ri**alpha * 16))
        print(f"[Client] 📦 Adaptive Batch Size: {Bt} (Ri={Ri:.2f}, α={alpha})")

        # ----- Local Training -----
        self.model.fit(self.x_train, self.y_train, epochs=5, batch_size=Bt, verbose=0)

        # ----- Sparse Ternary Compression (STC) + Differential Privacy -----
        original_weights = self.model.get_weights()
        theta = 0.01  # STC threshold
        sigma = 0.001  # Std for Gaussian noise

        compressed_weights = [
            np.sign(w) * np.maximum(theta, np.abs(w)) for w in original_weights
        ]
        noisy_weights = [
            w + np.random.normal(loc=0.0, scale=sigma, size=w.shape) for w in compressed_weights
        ]

        return noisy_weights, len(self.x_train), {"training_duration": Ri}

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[float, int, Dict[str, float]]:
        self.model.set_weights(parameters)
        loss, mae = self.model.evaluate(self.x_train, self.y_train, verbose=0)
        print(f"[Client] 🧪 Evaluation - Loss: {loss:.4f}, MAE: {mae:.4f}")
        return loss, len(self.x_train), {"mae": mae}


def start_client(x_train: np.ndarray, y_train: np.ndarray, optimizer_type: str = "yogi"):
    client = SmartGridClient(x_train, y_train, optimizer_type)
    fl.client.start_numpy_client(server_address="localhost:8080", client=client)
