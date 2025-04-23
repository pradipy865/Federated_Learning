import numpy as np
import threading
import argparse
import logging
import pickle
import time
import grpc
import flwr as fl

from client import SmartGridClient
from utils import load_and_preprocess_data, split_data_among_clients, load_test_data
from model import create_model
from sklearn.metrics import mean_squared_error, r2_score
from flwr.common import parameters_to_ndarrays  # ✅ Added for final evaluation

# --------------------------- Logging Configuration ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s",
    handlers=[
        logging.FileHandler("fl_training.log"),
        logging.StreamHandler()
    ]
)

# ---------------------------- Argument Parser -------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Federated Learning Simulation")
    parser.add_argument("--data", type=str, default="data/smart_grid_dataset.csv", help="Path to dataset CSV")
    parser.add_argument("--clients", type=int, default=5, help="Number of simulated clients")
    parser.add_argument("--optimizer", type=str, default="yogi", help="Optimizer type (e.g., 'adam', 'yogi')")
    parser.add_argument("--delay", type=int, default=5, help="Delay before starting clients (in seconds)")
    parser.add_argument("--interval", type=int, default=2, help="Interval between client startups (seconds)")
    return parser.parse_args()

# -------------------------- Client Simulation Thread --------------------------
def start_client(X_c: np.ndarray, y_c: np.ndarray, optimizer: str, client_id: int):
    try:
        logging.info(f"🚀 Client-{client_id} attempting to connect...")
        client = SmartGridClient(X_c, y_c, optimizer_type=optimizer)
        fl.client.start_client(server_address="localhost:8080", client=client)
        logging.info(f"✅ Client-{client_id} finished.")
    except grpc.RpcError as e:
        logging.error(f"❌ Client-{client_id} gRPC error: {e.code()} - {e.details()}")
    except Exception as ex:
        logging.exception(f"❌ Client-{client_id} unexpected exception: {ex}")

# --------------------------- Final Evaluation ---------------------------
def evaluate_global_model(model_path: str, optimizer: str, data_path: str):
    try:
        X_test, y_test = load_test_data(data_path)
        model = create_model(input_shape=X_test.shape[1], optimizer_type=optimizer)

        with open(model_path, "rb") as f:
            global_parameters = pickle.load(f)

        # ✅ Convert Parameters to NumPy weights
        weights = parameters_to_ndarrays(global_parameters)
        model.set_weights(weights)
        logging.info("✅ Loaded best global model weights.")

        y_pred = model.predict(X_test).flatten()
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print("\n📊 Final Centralized Evaluation:")
        print(f"🔹 Mean Squared Error (MSE): {mse:.4f}")
        print(f"🔹 R² Score: {r2:.4f}")

    except FileNotFoundError:
        logging.warning("⚠️ best_model_weights.pkl not found. Skipping evaluation.")
    except Exception as e:
        logging.exception("❌ Final evaluation failed")

# --------------------------- Main Function ---------------------------
def main():
    args = parse_args()

    # Step 1: Load and split data
    logging.info("📦 Loading and splitting dataset...")
    X_scaled, y, _ = load_and_preprocess_data(args.data)
    client_splits = split_data_among_clients(X_scaled, y, num_clients=args.clients)

    # Step 2: Delay before starting clients
    logging.info(f"⏳ Waiting {args.delay} seconds for server to be ready...")
    time.sleep(args.delay)

    # Step 3: Start clients sequentially with interval
    threads = []
    for i, (X_c, y_c) in enumerate(client_splits, start=1):
        thread = threading.Thread(
            target=start_client,
            args=(np.array(X_c), np.array(y_c), args.optimizer, i),
            name=f"ClientThread-{i}"
        )
        thread.start()
        threads.append(thread)
        time.sleep(args.interval)  # Stagger client start time

    # Step 4: Wait for all clients to complete
    for thread in threads:
        thread.join()

    logging.info("✅ All clients finished. Starting final evaluation...")
    evaluate_global_model("best_model_weights.pkl", args.optimizer, args.data)

# --------------------------- Entry Point ---------------------------
if __name__ == "__main__":
    main()
