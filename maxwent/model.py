import tensorflow as tf


class MaxWEnt(tf.keras.Model):
    
    def __init__(self, network, lambda_=1., n_pred=50, **kwargs):
        super(MaxWEnt, self).__init__()
        self.network = network
        self.lambda_ = lambda_
        self.n_pred = n_pred

        self.weight_entropy_metric = tf.keras.metrics.Mean(name="weight_entropy")


    def call(self, inputs, training=False):
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
        return self.network(inputs, training=training)

    
    def build(self, input_shape):
        if not self.network.built:
            self.network.build(input_shape)
            super(MaxWEnt, self).build(input_shape)
        else:
            super(MaxWEnt, self).build(self.network.input_shape)


    def fit_svd(self, x, batch_size=32):
        dummy = x[:1]
        data = tf.data.Dataset.from_tensor_slices(x).batch(batch_size)
        for layer in self.network.layers:
            if hasattr(layer, "fit_svd_"):
                layer.fit_svd_ = "start"
        self.network(dummy)
        for batch in data:
            self.network(batch)
        for layer in self.network.layers:
            if hasattr(layer, "fit_svd_"):
                layer.fit_svd_ = "end"
        self.network(dummy)
            