import tensorflow as tf
import numpy as np


class MaxWEnt(tf.keras.Model):
    """
    MaxWEnt (Maximum Weight Entropy) model, extending TensorFlow's Keras Model.
    This model applies a weight entropy regularization term to encourage diversity
    in the learned parameters of a neural network.
    
    Args:
        network (tf.keras.Model): The base pretrained neural network model.
        lambda_ (float): A regularization coefficient for controlling weight entropy.
            Large `lambda_` generally implies more weight entropy.
    """
    
    def __init__(self, network, lambda_=1.):
        super(MaxWEnt, self).__init__()
        self.network = network
        self.lambda_ = lambda_

        self.weight_entropy_metric = tf.keras.metrics.Mean(name="weight_entropy")

    def call(self, inputs, training=False, clip=None, seed=None):
        """
        Performs a forward pass through the network.

        Args:
            inputs (tf.Tensor): Input data.
            training (bool, optional): Whether the model is in training mode. Defaults to False.
            clip (float, optional): Clipping value to use on the weight variance.
                If `clip` is `None`, no cliping is applied. If `clip = 0` there is no weight variance.
            seed (int, optional): Set the random seed in the layers. It is useful to use the same sampled
                network accross multiple batch of data.

        Returns:
            tf.Tensor: The output of the network.
        """
        if clip is not None:
            self._update_clip_in_layers(clip)
        if seed is not None:
            self._update_seed_in_layers(seed)
        if training:
            weight_loss = 0.
            num_params = 0.
            for weight in self.trainable_variables:
                if "maxwent" in weight.name:
                    w = tf.math.softplus(weight)
                    weight_loss += tf.reduce_sum(w)
                    num_params += tf.cast(tf.reduce_prod(w.shape), dtype=w.dtype)
            weight_loss /= num_params
            weight_loss *= -self.lambda_
            self.add_loss(weight_loss)
            self.weight_entropy_metric.update_state(weight_loss)

        out = self.network(inputs, training=training)
        if clip is not None:
            self._update_clip_in_layers(None)
        if seed is not None:
            self._update_seed_in_layers(None)
        return out

    def _update_clip_in_layers(self, clip):
        """
        Updates the clipping value for all layers that support it.

        Args:
            clip (float or None): Clipping value to be applied to layers.
        """
        for layer in self.network.layers:
            if hasattr(layer, "clip_"):
                setattr(layer, "clip_", clip)

    def _update_seed_in_layers(self, seed):
        """
        Updates the seed value for all layers that support it.

        Args:
            seed (int or None): Seed value to be applied to layers.
        """
        for layer in self.network.layers:
            if hasattr(layer, "seed_"):
                setattr(layer, "seed_", seed)

    def build(self, input_shape):
        """
        Builds the model by initializing the underlying network if it hasn't been built yet.

        Args:
            input_shape (tuple): The shape of the input data.
        """
        if not self.network.built:
            self.network.build(input_shape)
            super(MaxWEnt, self).build(input_shape)
        else:
            super(MaxWEnt, self).build(self.network.input_shape)

    def fit_svd(self, x, batch_size=32):
        """
        Fit the Singular Value Decomposition (SVD) transition matrix on specific
        layers of the network.

        Args:
            x (np.array or tf.data.Dataset): The input data.
            batch_size (int, optional): Batch size for processing. Defaults to 32.
        """
        if not isinstance(x, tf.data.Dataset):
            num_steps = int(np.ceil(len(x) / batch_size))
            data = tf.data.Dataset.from_tensor_slices(x).batch(batch_size)
        else:
            num_steps = None
            data = x
        dummy = next(iter(data))
        for layer in self.network.layers:
            if hasattr(layer, "fit_svd_"):
                layer.fit_svd_ = "start"
        self.network(dummy, training=False)
        step = 0
        for batch in data:
            self.network(batch, training=False)
            step += 1
            if num_steps is not None and step == num_steps:
                break
        for layer in self.network.layers:
            if hasattr(layer, "fit_svd_"):
                layer.fit_svd_ = "end"
        self.network(dummy, training=False)

    def predict(self, x, batch_size=32, clip=None, seed=None):
        """
        Makes predictions on input data.

        Args:
            x (np.array or tf.data.Dataset): The input data.
            batch_size (int, optional): Batch size for processing. Defaults to 32.
            clip (float, optional): Clipping value to use on the weight variance.
                If `clip` is `None`, no cliping is applied. If `clip = 0` there is no weight variance.
            seed (int, optional): Set the random seed in the layers. It is useful to use the same sampled
                network accross multiple batch of data.

        Returns:
            np.array: Predictions as a NumPy array.
        """
        if not isinstance(x, tf.data.Dataset):
            num_steps = int(np.ceil(len(x) / batch_size))
            data = tf.data.Dataset.from_tensor_slices(x).batch(batch_size)
        else:
            num_steps = None
            data = x
        outputs = []
        step = 0
        for batch in data:
            out = self.call(batch, training=False, clip=clip, seed=seed)
            outputs.append(out)
            step += 1
            if num_steps is not None and step == num_steps:
                break
        return tf.concat(outputs, axis=0).numpy()

    def predict_mean(self, x, batch_size=32, clip=0., n_sample=1):
        """
        Computes the mean prediction over multiple stochastic forward passes.

        Args:
            x (np.array or tf.data.Dataset): The input data.
            batch_size (int, optional): Batch size for processing. Defaults to 32.
            clip (float, optional): Clipping value to use on the weight variance.
                If `clip` is `None`, no cliping is applied. If `clip = 0` there is no weight variance.
            n_sample (int, optional): Number of stochastic forward passes. Defaults to 1.

        Returns:
            np.array: The mean prediction.
        """
        preds = []
        kwargs = dict(batch_size=batch_size, clip=clip, seed=None)
        seeds = tf.random.uniform(shape=(n_sample,),
                                  maxval=10**10,
                                  dtype=tf.int32).numpy()
        for i in range(n_sample):
            kwargs["seed"] = int(seeds[i])
            preds.append(self.predict(x, **kwargs))
        preds = tf.stack(preds, axis=-1)
        pred_mean = tf.reduce_mean(preds, axis=-1).numpy()
        return pred_mean

    def predict_std(self, x, batch_size=32, clip=None, n_sample=10):
        """
        Computes the standard deviation of predictions over multiple stochastic forward passes.

        Args:
            x (np.array or tf.data.Dataset): The input data.
            batch_size (int, optional): Batch size for processing. Defaults to 32.
            clip (float, optional): Clipping value to use on the weight variance.
                If `clip` is `None`, no cliping is applied. If `clip = 0` there is no weight variance.
            n_sample (int, optional): Number of stochastic forward passes. Defaults to 10.

        Returns:
            np.array: The standard deviation of the predictions.
        """
        preds = []
        kwargs = dict(batch_size=batch_size, clip=clip, seed=None)
        seeds = tf.random.uniform(shape=(n_sample,),
                                  maxval=10**10,
                                  dtype=tf.int32).numpy()
        for i in range(n_sample):
            kwargs["seed"] = int(seeds[i])
            preds.append(self.predict(x, **kwargs))
        preds = tf.stack(preds, axis=-1)
        pred_std = tf.math.reduce_std(preds, axis=-1).numpy()
        return pred_std