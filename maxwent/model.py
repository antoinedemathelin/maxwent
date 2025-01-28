import tensorflow as tf


class MaxWEnt(tf.keras.Model):
    
    def __init__(self, network, lambda_=1., n_pred=50, **kwargs):
        super(MaxWEnt, self).__init__()
        self.network = network
        self.lambda_ = lambda_
        self.n_pred = n_pred

        self.weight_entropy_metric = tf.keras.metrics.Mean(name="weight_entropy")


    def call(self, inputs, training=False, clip=None, seed=None):
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
        for layer in self.network.layers:
            if hasattr(layer, "clip_"):
                setattr(layer, "clip_", clip)


    def _update_seed_in_layers(self, seed):
        for layer in self.network.layers:
            if hasattr(layer, "seed_"):
                setattr(layer, "seed_", seed)

    
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
        self.network(dummy, training=False)
        for batch in data:
            self.network(batch, training=False)
        for layer in self.network.layers:
            if hasattr(layer, "fit_svd_"):
                layer.fit_svd_ = "end"
        self.network(dummy, training=False)


    def predict(self, x, batch_size=32, clip=None, seed=None):
        if not isinstance(x, tf.data.Dataset):
            data = tf.data.Dataset.from_tensor_slices(x).batch(batch_size)
        else:
            data = x
        outputs = []
        for batch in data:
            out = self.call(batch, training=False, clip=clip, seed=seed)
            outputs.append(out)
        return tf.concat(outputs, axis=0).numpy()


    def predict_mean(self, x, batch_size=32, clip=0., n_sample=1):
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
