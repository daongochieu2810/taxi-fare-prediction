import os
from spark_preprocessing import preprocess_data
from mlp_net import MLP


def main():
    # params for preprocessing and learning
    filepath = os.path.dirname(os.path.realpath(__file__))
    params = {
        "data_dir": os.path.join(
            filepath, "data"
        ),  # directory where taxi data csv files are stored
        "save_dir": os.path.join(
            filepath, "results"
        ),  # directory for storing processed data, predictions and plots
        "epochs": 5,
        "batch_size": 64,
        "learning_rate": 0.001,
        "hidden_units": 32,  # number of units for each MLP layer
    }

    if not os.path.exists(params["save_dir"]):
        os.makedirs(os.path.abspath(params["save_dir"]))

    # spark job to preprocess data
    preprocess_data(params=params)

    # distributed training and inference
    mlp_model = MLP(params=params)
    mlp_model.train_model()
    mlp_model.model_inference()
    mlp_model.calculate_rmse()
    print("MLP training and model inference done!")


if __name__ == "__main__":
    main()
