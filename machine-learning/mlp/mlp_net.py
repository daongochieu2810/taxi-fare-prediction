import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.metrics import RootMeanSquaredError
from tensorflow.python.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    CSVLogger,
)

# fix random seed for reproducibility
tf.random.set_seed(42)


class MLP(Model):
    def __init__(self, params):
        super(MLP, self).__init__()

        self.params = params
        self.save_dir = params["save_dir"]
        self.epochs = params["epochs"]
        self.batch_size = params["batch_size"]
        self.learning_rate = params["learning_rate"]
        self.hidden_units = params["hidden_units"]
        self.checkpoint_filepath = os.path.join(
            self.save_dir, "MLP_checkpoints_keras/chk"
        )

        # read train and test data
        df_train = pd.read_parquet(os.path.join(self.save_dir, "train_data"))
        df_test = pd.read_parquet(os.path.join(self.save_dir, "test_data"))

        df_train_features = df_train["scaled_features"].apply(pd.Series)
        df_test_features = df_test["scaled_features"].apply(pd.Series)
        print(df_train_features.head())

        # convert df into numpy array
        self.train_x = np.array(df_train_features["values"].to_numpy().tolist())
        self.train_y = df_train["fare"].to_numpy()
        self.test_x = np.array(df_test_features["values"].to_numpy().tolist())
        self.test_y = df_test["fare"].to_numpy()

        self.build_MLP()

        print("---------------------------")
        print("| LEARNING CONFIGURATION  |")
        print("---------------------------")
        print("self.train_x.shape = ", self.train_x.shape)
        print("self.train_y.shape = ", self.train_y.shape)
        print("self.test_x.shape = ", self.test_x.shape)
        print("self.test_y.shape = ", self.test_y.shape)
        print("---------------------------")

    def build_MLP(self):
        # build model
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            self.model = Sequential(
                [
                    Dense(
                        self.hidden_units,
                        input_shape=(self.train_x.shape[1],),
                        activation="relu",
                    ),
                    Dense(self.hidden_units, activation="relu"),
                    Dense(self.hidden_units, activation="relu"),
                    Dense(1, activation="relu"),
                ]
            )
            self.model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))

    def train_model(self):
        # train model
        print("MODEL SUMMARY: \n", self.model.summary())
        csv_logger = CSVLogger(
            os.path.join(self.save_dir, "train_history.csv"),
            separator=",",
            append=False,
        )
        cb_chk = ModelCheckpoint(
            monitor="val_loss", save_best_only=True, filepath=self.checkpoint_filepath
        )
        cb_early = EarlyStopping(monitor="val_loss", patience=10, verbose=1)
        cb_lr = ReduceLROnPlateau(monitor="val_loss", patience=3, verbose=0, factor=0.2)
        history = self.model.fit(
            x=self.train_x,
            y=self.train_y,
            callbacks=[cb_chk, cb_early, cb_lr, csv_logger],
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
        )
        print("- end_training")
        self.plot_history(history.history)

    def model_inference(self):
        # make predictions on test data
        print("Model inference")
        reconstructed_model = tf.keras.models.load_model(self.checkpoint_filepath)
        print("Model inference complete")

        predictions = reconstructed_model.predict(self.test_x)
        np.save(os.path.join(self.save_dir, "predictions.npy"), predictions)

    def calculate_rmse(self):
        # calculate RMSE
        print("Calculating RMSE")
        rmse = RootMeanSquaredError()
        predictions = np.load(os.path.join(self.save_dir, "predictions.npy"))
        rmse.update_state(y_pred=predictions, y_true=self.test_y)
        print("RMSE: " + str(rmse.result().numpy()))

    def plot_history(self, history):
        # plot training loss
        plt.figure()
        plt.plot(history["loss"], "-k", label="training loss")
        plt.plot(history["val_loss"], "--b", label="validation loss")
        plt.xlabel("epochs")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "loss.png"), dpi=200)
