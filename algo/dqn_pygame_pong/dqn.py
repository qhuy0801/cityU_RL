"""
This module offers a neural network implementation utilising the Keras library.
Please note that the `GaussianNoise` layer is for noisy network design.
"""
from keras.models import Sequential
from keras.layers import Dense, GaussianNoise


class DQN:
    """
    This class uses Keras to create a neural network with an input layer, two hidden layers with a noise layer in between, and an output layer.
    """
    def __init__(self, _state_count, _action_count):
        self.state_count = _state_count
        self.action_count = _action_count
        self.model = self.compile_net()

    def compile_net(self):
        """
        Compile the network
        :return: network model
        """
        model = Sequential()
        model.add(Dense(units=64, activation="relu", input_dim=self.state_count))
        model.add(GaussianNoise(0.1))
        model.add(Dense(units=32, activation="relu"))
        model.add(Dense(units=self.action_count, activation="linear"))
        model.compile(loss="mse", optimizer="adam")
        return model

    def _fit(self, _x, _y, _epoch=1, _verbose=0):
        """
        Train the network
        :param _x: input
        :param _y: output
        :param _epoch:
        :param _verbose:
        :return: None
        """
        self.model.fit(_x, _y, batch_size=64, epochs=_epoch, verbose=_verbose)

    def _predict(self, x):
        """
        Use the network to predict one batch
        :param x:
        :return: predicted result
        """
        return self.model.predict(x)

    def _predict_single(self, _x):
        """
        Use the network to predict single input
        :param _x:
        :return: predicted result
        """
        x = _x.reshape(1, self.state_count)
        x = self._predict(x).flatten()
        return x
