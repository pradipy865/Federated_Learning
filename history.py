import pickle

# Load history from the pickle file
with open("history.pkl", "rb") as f:
    history = pickle.load(f)

# Display the contents
print("=== Training History ===\n")

print("Distributed Losses:")
for rnd, loss in history.get("loss_distributed", []):
    print(f"Round {rnd}: Loss = {loss:.4f}")

print("\nCentralized Losses:")
for rnd, loss in history.get("loss_centralized", []):
    print(f"Round {rnd}: Loss = {loss:.4f}")

print("\nDistributed Metrics:")
for rnd, metrics in history.get("metrics_distributed", []):
    print(f"Round {rnd}: {metrics}")

print("\nCentralized Metrics:")
for rnd, metrics in history.get("metrics_centralized", []):
    print(f"Round {rnd}: {metrics}")
