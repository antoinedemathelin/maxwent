import tensorflow as tf


class DropoutOff(tf.keras.layers.Dropout):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dropout_off_ = True
    
    def call(self, inputs, training=False):
        if self.dropout_off_:
            return inputs
        else:
            return super().call(inputs, training=training)

    def set_dropout_off(self, dropout_off: bool):
        self.dropout_off_ = dropout_off


class SpatialDropout1DOff(tf.keras.layers.SpatialDropout1D):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dropout_off_ = True
    
    def call(self, inputs, training=False):
        if self.dropout_off_:
            return inputs
        else:
            return super().call(inputs, training=training)

    def set_dropout_off(self, dropout_off: bool):
        self.dropout_off_ = dropout_off


class SpatialDropout2DOff(tf.keras.layers.SpatialDropout2D):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dropout_off_ = True
    
    def call(self, inputs, training=False):
        if self.dropout_off_:
            return inputs
        else:
            return super().call(inputs, training=training)

    def set_dropout_off(self, dropout_off: bool):
        self.dropout_off_ = dropout_off


class SpatialDropout3DOff(tf.keras.layers.SpatialDropout3D):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dropout_off_ = True
    
    def call(self, inputs, training=False):
        if self.dropout_off_:
            return inputs
        else:
            return super().call(inputs, training=training)

    def set_dropout_off(self, dropout_off: bool):
        self.dropout_off_ = dropout_off


class DenseMaxWEnt(tf.keras.layers.Dense):
    
    def __init__(
        self,
        kernel_distrib="uniform",
        bias_distrib="uniform",
        kernel_var_init=-7.,
        bias_var_init=-7.,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.kernel_distrib = kernel_distrib
        self.bias_distrib = bias_distrib
        self.kernel_var_init = kernel_var_init
        self.bias_var_init = bias_var_init
        self.use_svd_ = False
        self.fit_svd_ = None
        self.clip_ = None
    
    
    def build(self, input_shape):

        last_dim = input_shape[-1]

        self.kernel_var_initializer = tf.keras.initializers.Constant(
                value=self.kernel_var_init)
        
        self.maxwent_kernel = self.add_weight(
            name="maxwent_kernel",
            shape=[last_dim, self.units],
            initializer=self.kernel_var_initializer,
            dtype=self.dtype,
            trainable=True,
        )

        if self.use_bias:
            self.bias_var_initializer = tf.keras.initializers.Constant(
                value=self.bias_var_init)
            
            self.maxwent_bias = self.add_weight(
                name="maxwent_bias",
                shape=[self.units,],
                initializer=self.bias_var_initializer,
                dtype=self.dtype,
                trainable=True,
            )
        
        self.maxwent_Vmatrix = self.add_weight(
            name="maxwent_Vmatrix",
            shape=[last_dim, last_dim],
            initializer="zeros",
            dtype=self.dtype,
            trainable=False,
        )
        self.maxwent_Vmatrix.assign(tf.eye(last_dim))

        super().build(input_shape)
    
    
    def _z_sample(self, kind, shape):
        if kind == "normal":
            z = tf.random.normal(shape)
        elif kind == "uniform":
            z = tf.random.uniform(shape) * 2. - 1.
        elif kind == "bernoulli":
            z = tf.random.uniform(shape)
            z = tf.cast(tf.math.greater(0.5, z), self.kernel.dtype) * 2. - 1.
        else:
            raise ValueError("Unknow noise distribution")
        return z
    
    
    def call(self, inputs):
        if self.fit_svd_:
            self.fit_svd(inputs, mode=self.fit_svd_)
            kernel = self.kernel
            bias = self.bias
        else:
            z = self._z_sample(self.kernel_distrib, self.maxwent_kernel.shape)
            z = tf.math.softplus(self.maxwent_kernel) * z
            if self.clip_ is not None:
                z = tf.clip_by_value(z, -self.clip_, self.clip_)
            if self.use_svd_ is not None:
                z = tf.matmul(self.maxwent_Vmatrix, z)
            kernel = self.kernel + z
    
            if self.bias is not None:
                z = self._z_sample(self.bias_distrib, self.maxwent_bias.shape)
                z = tf.math.softplus(self.maxwent_bias) * z
                if self.clip_ is not None:
                    z = tf.clip_by_value(z, -self.clip_, self.clip_)
                bias = self.bias + z
            
        x = tf.matmul(inputs, kernel)
        if self.bias is not None:
            x = tf.add(x, bias)
        if self.activation is not None:
            x = self.activation(x)
        return x


    def fit_svd(self, inputs, mode="train"):
        if mode == "start":
            self.XTX_ = tf.zeros(self.maxwent_Vmatrix.shape)
            self.fit_svd_ = "train"
        elif mode == "train":
            self.XTX_ += tf.matmul(tf.transpose(inputs), inputs)
        elif mode == "end":
            _, V = tf.linalg.eig(self.XTX_)
            V = tf.math.real(V)
            V = tf.cast(V, dtype=self.maxwent_Vmatrix.dtype)
            self.maxwent_Vmatrix.assign(V)
            self.fit_svd_ = None
            self.use_svd_ = True
            delattr(self, "XTX_")
        else:
            raise ValueError("mode should be in [start, train, end]")


    def save_own_variables(self, store):
        super().save_own_variables(store)
        target_variables = [self.maxwent_kernel, self.maxwent_Vmatrix]
        if self.use_bias:
            target_variables.append(self.maxwent_bias)
        for i, variable in enumerate(target_variables):
            store["mwe%i"%i] = variable


    def load_own_variables(self, store):
        super().load_own_variables(store)
        target_variables = [self.maxwent_kernel, self.maxwent_Vmatrix]
        if self.use_bias:
            target_variables.append(self.maxwent_bias)
        for i, variable in enumerate(target_variables):
            variable.assign(store["mwe%i"%i])


    def get_config(self):
        base_config = super().get_config()
        config = dict(
            kernel_distrib=self.kernel_distrib,
            bias_distrib=self.bias_distrib,
            kernel_var_init=self.kernel_var_init,
            bias_var_init=self.bias_var_init,
        )
        base_config.update(config)
        return base_config