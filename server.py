import os
import math
import pickle
import logging
import numpy as np
from typing import List, Tuple, Optional, Dict, Union

import flwr as fl
from flwr.server.strategy import Strategy
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays, Scalar, Parameters, FitRes, EvaluateRes

from model import create_model
from utils import load_test_data
from sklearn.metrics import mean_squared_error, r2_score

# ---------- Logging Setup ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ---------- Global Parameters ----------
latest_global_parameters: Optional[Parameters] = None
best_model_parameters: Optional[Parameters] = None
best_r2 = float("-inf")

# ---------- Metric Aggregation ----------
def weighted_average(metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, float]:
    total_examples = sum(num_examples for num_examples, _ in metrics)
    mse_sum = sum(m.get("mse", 0.0) * num_examples for num_examples, m in metrics)
    r2_sum = sum(m.get("r2", 0.0) * num_examples for num_examples, m in metrics)
    return {
        "mse": mse_sum / total_examples if total_examples > 0 else 0.0,
        "r2": r2_sum / total_examples if total_examples > 0 else 0.0,
    }

# ---------- Central Evaluation ----------
def get_evaluate_fn():
    X_test, y_test = load_test_data("data/smart_grid_dataset.csv")

    def evaluate(server_round: int, parameters: Parameters, config: Dict[str, Scalar]):
        global latest_global_parameters, best_model_parameters, best_r2

        model = create_model(input_shape=X_test.shape[1])
        model.set_weights(parameters_to_ndarrays(parameters))

        y_pred = model.predict(X_test).flatten()
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        logging.info(f"[Round {server_round}] Evaluation - MSE: {mse:.4f}, R²: {r2:.4f}")

        if r2 > best_r2:
            best_r2 = r2
            best_model_parameters = parameters
            model.save("best_model.keras")
            logging.info(f"🌟 New best model at round {server_round} with R² = {r2:.4f}")

        latest_global_parameters = parameters
        return mse, {"mse": mse, "r2": r2, "loss": mse}

    return evaluate

# ---------- Custom Async FedAvg Strategy ----------
class AsyncFedAvg(Strategy):
    def __init__(self, initial_parameters: Parameters):
        self.parameters = initial_parameters
        self.evaluate_fn = get_evaluate_fn()

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        return self.parameters

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, fl.common.FitIns]]:
        sample_size = client_manager.num_available()
        num_clients = max(2, min(math.ceil(0.5 * sample_size), sample_size))
        sampled_clients = client_manager.sample(num_clients, min_num_clients=2)
        config = {"server_round": server_round}
        return [(client, fl.common.FitIns(parameters, config)) for client in sampled_clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        global latest_global_parameters

        if not results:
            logging.warning(f"[Round {server_round}] No successful fit results.")
            return None, {}

        if failures:
            logging.warning(f"[Round {server_round}] {len(failures)} client failures occurred.")

        current_weights = parameters_to_ndarrays(self.parameters)
        aggregated_update = [np.zeros_like(w) for w in current_weights]
        total_weight = 0.0

        for _, fit_res in results:
            duration = fit_res.metrics.get("training_duration", 1.0)
            client_weights = parameters_to_ndarrays(fit_res.parameters)
            for i in range(len(current_weights)):
                aggregated_update[i] += duration * (client_weights[i] - current_weights[i])
            total_weight += duration

        if total_weight > 0:
            updated_weights = [
                w + delta / total_weight for w, delta in zip(current_weights, aggregated_update)
            ]
            self.parameters = ndarrays_to_parameters(updated_weights)
            latest_global_parameters = self.parameters

        return self.parameters, {"aggregated_round": server_round}

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, fl.common.EvaluateIns]]:
        sample_size = client_manager.num_available()
        num_clients = max(1, min(math.ceil(0.1 * sample_size), sample_size))
        sampled_clients = client_manager.sample(num_clients, min_num_clients=1)
        config = {"server_round": server_round}
        return [(client, fl.common.EvaluateIns(parameters, config)) for client in sampled_clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]]
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if not results:
            logging.warning(f"[Round {server_round}] No evaluation results.")
            return None, {}

        if failures:
            logging.warning(f"[Round {server_round}] {len(failures)} evaluation failures occurred.")

        losses = [res.metrics.get("loss", 0.0) for _, res in results]
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        return avg_loss, {"avg_loss": avg_loss}

    def evaluate(self, server_round: int, parameters: Parameters):
        return self.evaluate_fn(server_round, parameters, {})

# ---------- Server Entry Point ----------
def start_server():
    logging.info("🚀 Starting Federated Server...")

    X_test, _ = load_test_data("data/smart_grid_dataset.csv")
    model = create_model(input_shape=X_test.shape[1])
    initial_parameters = ndarrays_to_parameters(model.get_weights())

    strategy = AsyncFedAvg(initial_parameters=initial_parameters)
    history = fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy
    )

    # Save models
    if latest_global_parameters is not None:
        with open("global_model_weights.pkl", "wb") as f:
            pickle.dump(latest_global_parameters, f)
        logging.info("✅ Saved final global model weights to 'global_model_weights.pkl'")
    else:
        logging.warning("⚠️ Final model weights not available")

    if best_model_parameters is not None:
        with open("best_model_weights.pkl", "wb") as f:
            pickle.dump(best_model_parameters, f)
        logging.info("🌟 Saved best performing model weights to 'best_model_weights.pkl'")
    else:
        logging.warning("⚠️ Best model weights not available")

    try:
        extracted_history = {
            "loss_distributed": history.losses_distributed,
            "loss_centralized": history.losses_centralized,
            "metrics_distributed": history.metrics_distributed,
            "metrics_centralized": history.metrics_centralized,
        }
        with open("history.pkl", "wb") as f:
            pickle.dump(extracted_history, f)
        logging.info("📈 Saved training history to 'history.pkl'")
    except Exception as e:
        logging.error(f"⚠️ Could not save training history: {e}")

if __name__ == "__main__":
    start_server()
