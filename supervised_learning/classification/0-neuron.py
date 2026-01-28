#!/usr/bin/env python3
"""
This module defines the Neuron class for binary classification.
"""
import numpy as np


class Neuron:
    """
    Defines a single neuron performing binary classification.
    """

    def __init__(self, nx):
        """
        Class constructor.
        Args:
            nx (int): The number of input features to the neuron.
        Raises:
            TypeError: If nx is not an integer.
            ValueError: If nx is less than 1.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Initializing Weights as a row vector of shape (1, nx)
        self.W = np.random.randn(1, nx)
        # Initializing Bias to 0
        self.b = 0
        # Initializing Activated output to 0
        self.A = 0