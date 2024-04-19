import flwr as fl
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Simulated client data (replace with your actual data access logic)
clients = [
    ("client1", np.random.rand(10, 5), np.random.randint(0, 2, size=10)),  # Features, Labels
    ("client2", np.random.rand(15, 5), np.random.randint(0, 2, size=15)),
    # ... add more clients with their data
]


# Define functions for local training on clients
def get_local_model(params):
    # Initialize local SVM model with received parameters (if any)
    model = SVC(kernel='linear', gamma='auto', C=1)
    if params:
        model.set_params(**params)
    return model

def fit_local_model(model, data):
    # Extract features (X) and labels (y) from data
    X, y = data

    # Split data into local training and validation sets (optional)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the local model
    model.fit(X_train, y_train)

    # Evaluate the local model on validation set (optional)
    # ... (add evaluation logic using metrics like accuracy)

    # Return updated model parameters
    return model.get_params()


# Define federated learning function (server-side)
def federated_learning(num_rounds):
    # Define model strategy (FedAvg)
    strategy = fl.server.strategy.FedAvg(fraction=1.0)  # Adjust fraction as needed

    # Create a Flower server instance
    server = fl.server.Server(strategy=strategy)

    # Configure server with local model function
    def model_fn():
        return get_local_model(params={})

    server.register(config={"num_rounds": num_rounds}, model=model_fn)

    # Simulate client selection (replace with actual selection logic)
    def client_selection(num_clients):
        return [clients[i] for i in np.random.choice(len(clients), num_clients, replace=False)]

    # Federated training loop
    for round in range(num_rounds):
        # Select clients for participation in this round
        selected_clients = client_selection(num_clients=min(len(clients), 5))  # Adjust max clients

        # Fit the model on federated data (simulated exchange)
        results = server.fit(
            {(client_id, data) for client_id, data in selected_clients}, timeout=180
        )

        # Access the aggregated model parameters
        global_model_params = results.model.get_params()

        print(f"Round {round+1}: Completed with aggregated parameters")

    return global_model_params


# Run federated learning
global_model_params = federated_learning(num_rounds=5)

# Use the global_model_params to create a final model on the server (optional)
final_model = get_local_model(global_model_params)

# ... (use final_model for further tasks like evaluation or predictions)