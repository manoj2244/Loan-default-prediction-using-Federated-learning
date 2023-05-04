import warnings
import flwr as fl
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import grpc

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

import utils

if __name__ == "__main__":
    data =  pd.read_csv("Fin.csv")
    data.drop(["Index"] , inplace =  True , axis = 1)
    X = data.drop("Defaulted?" , axis = 1)
    Y = data["Defaulted?"]
    from sklearn.model_selection import train_test_split
     

    X_train , X_test , y_train , y_test = train_test_split(X , Y , test_size = 0.3)


    partition_id = np.random.choice(2)
    (X_train, y_train) = utils.partition(X_train, y_train, 2)[partition_id]


    # Create LogisticRegression Model
    model = LogisticRegression(
        penalty="l2",
        max_iter=1,  # local epoch
        warm_start=True,  # prevent refreshing weights when fitting
    )

    # Setting initial parameters, akin to model.compile for keras models
    utils.set_initial_params(model)

    # Define Flower client
    class MnistClient(fl.client.NumPyClient):
        def get_parameters(self, config):  # type: ignore
            return utils.get_model_parameters(model)

        def fit(self, parameters, config):  # type: ignore
            utils.set_model_params(model, parameters)
            # Ignore convergence failure due to low local epochs
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
            print(f"Training finished for round {config['server_round']}")
            return utils.get_model_parameters(model), len(X_train), {}

        def evaluate(self, parameters, config):  # type: ignore
            utils.set_model_params(model, parameters)
            loss = log_loss(y_test, model.predict_proba(X_test))
            accuracy = model.score(X_test, y_test)
            return loss, len(X_test), {"accuracy": accuracy}
    
    server = "192.168.179.237:8080"
    channel = grpc.insecure_channel(server)
    # Start Flower client
    fl.client.start_numpy_client(server_address=server, client=MnistClient())
    print("Dipesh")
