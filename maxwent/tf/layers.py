import tensorflow as tf
from keras.src.layers.convolutional.base_conv import BaseConv


class DropoutOff(tf.keras.layers.Dropout):
    """
    A custom Dropout layer that allows disabling dropout during training.
    """
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
    """
    A custom Dropout layer that allows disabling dropout during training.
    """
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
    """
    A custom Dropout layer that allows disabling dropout during training.
    """
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
    """
    A custom Dropout layer that allows disabling dropout during training.
    """
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
    """
    A custom stochastic Dense layer. 
    This layer introduces additional learnable noise distributions for the 
    kernel and bias.

    Note that only the variance parameters are trainable.
    The mean kernel and bias are non-trainable.
    
    Args:
        kernel_distrib (str): Distribution type for the kernel noise ('uniform', 'normal', or 'bernoulli').
        bias_distrib (str): Distribution type for the bias noise ('uniform', 'normal', or 'bernoulli').
        kernel_var_init (float): Initial value for the kernel variance parameters (Default -7).
            Note that the kernel variance is equal to `tf.math.softplus(kernel_var_init)`.
        bias_var_init (float): Initial value for the bias variance parameters (Default -7).
            Note that the bias variance is equal to `tf.math.softplus(bias_var_init)`.

    Attributes:
        use_svd_ (bool): Flag indicating whether SVD parametirization should be applied.
        fit_svd_ (str or None): Mode indicating the SVD fitting phase ('start', 'train', 'end').
        clip_ (float or None): Clipping threshold for noise values.
        seed_ (int or None): Random seed for reproducibility.
    """

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
        self.seed_ = None
    
    def build(self, input_shape):
        """
        Builds the layer, initializing weights and additional MaxWEnt parameters.

        Args:
            input_shape (tuple): Shape of the input tensor.
        """
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
        """
        Generates a random noise sample based on the specified distribution type.

        Args:
            kind (str): The type of noise distribution ('normal', 'uniform', 'bernoulli').
            shape (tuple): The shape of the output noise tensor.

        Returns:
            tf.Tensor: A tensor containing sampled noise values.
        """
        if self.seed_ is None:
            rng = tf.random
        else:
            rng = tf.random.Generator.from_seed(self.seed_)
        
        if kind == "normal":
            z = rng.normal(shape)
        elif kind == "uniform":
            z = rng.uniform(shape) * 2. - 1.
        elif kind == "bernoulli":
            z = rng.uniform(shape)
            z = tf.cast(tf.math.greater(0.5, z), self.kernel.dtype) * 2. - 1.
        else:
            raise ValueError("Unknown noise distribution")
        return z
    
    def call(self, inputs):
        """
        Performs the forward pass, applying noise to kernel and bias.

        Args:
            inputs (tf.Tensor): Input tensor to the layer.

        Returns:
            tf.Tensor: The processed output tensor.
        """
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
        """
        Fit the Singular Value Decomposition (SVD) transition matrix on the input data.

        Args:
            inputs (tf.Tensor): Input data tensor.
            mode (str): The phase of SVD fitting ('start', 'train', 'end').
        """
        if mode == "start":
            self.XTX_ = tf.zeros(self.maxwent_Vmatrix.shape)
            self.fit_svd_ = "train"
            self.n_ = 0
        elif mode == "train":
            self.XTX_ += tf.matmul(tf.transpose(inputs), inputs)
            self.n_ += inputs.shape[0]
        elif mode == "end":
            self.XTX_ /= tf.cast(self.n_, self.XTX_.dtype)
            _, V = tf.linalg.eig(self.XTX_)
            V = tf.math.real(V)
            V = tf.cast(V, dtype=self.maxwent_Vmatrix.dtype)
            self.maxwent_Vmatrix.assign(V)
            self.fit_svd_ = None
            self.use_svd_ = True
            delattr(self, "XTX_")
            delattr(self, "n_")
        else:
            raise ValueError("mode should be in [start, train, end]")

    def save_own_variables(self, store):
        """
        Saves layer variables to a storage dictionary.

        Args:
            store (dict): Dictionary where variables will be stored.
        """
        super().save_own_variables(store)
        target_variables = [self.maxwent_kernel, self.maxwent_Vmatrix]
        if self.use_bias:
            target_variables.append(self.maxwent_bias)
        for i, variable in enumerate(target_variables):
            store["mwe%i" % i] = variable

    def load_own_variables(self, store):
        """
        Loads layer variables from a storage dictionary.

        Args:
            store (dict): Dictionary containing stored variables.
        """
        super().load_own_variables(store)
        target_variables = [self.maxwent_kernel, self.maxwent_Vmatrix]
        if self.use_bias:
            target_variables.append(self.maxwent_bias)
        for i, variable in enumerate(target_variables):
            variable.assign(store["mwe%i" % i])

    def get_config(self):
        """
        Returns the configuration of the layer for serialization.

        Returns:
            dict: A dictionary containing the layer configuration.
        """
        base_config = super().get_config()
        config = dict(
            kernel_distrib=self.kernel_distrib,
            bias_distrib=self.bias_distrib,
            kernel_var_init=self.kernel_var_init,
            bias_var_init=self.bias_var_init,
        )
        base_config.update(config)
        return base_config


class BaseConvMaxWEnt(BaseConv):
    """
    A custom stochastic convolutional layer.
    This layer introduces additional learnable noise distributions for the kernel and bias.

    Note that only the variance parameters are trainable.
    The mean kernel and bias are non-trainable.

    Args:
        kernel_distrib (str): Distribution type for the kernel noise ('uniform', 'normal', or 'bernoulli').
        bias_distrib (str): Distribution type for the bias noise ('uniform', 'normal', or 'bernoulli').
        kernel_var_init (float): Initial value for the kernel variance parameters (Default -7).
            Note that the kernel variance is equal to `tf.math.softplus(kernel_var_init)`.
        bias_var_init (float): Initial value for the bias variance parameters (Default -7).
            Note that the bias variance is equal to `tf.math.softplus(bias_var_init)`.

    Attributes:
        use_svd_ (bool): Flag indicating whether SVD parametirization should be applied.
        fit_svd_ (str or None): Mode indicating the SVD fitting phase ('start', 'train', 'end').
        clip_ (float or None): Clipping threshold for noise values.
        seed_ (int or None): Random seed for reproducibility.
    """

    def build(self, input_shape):
        """
        Builds the layer, initializing MaxWEnt parameters, including kernel and bias noise.

        Args:
            input_shape (tuple): Shape of the input tensor.
        """
        super().build(input_shape)
        
        if self.data_format == "channels_last":
            input_channel = input_shape[-1]
        else:
            input_channel = input_shape[1]

        kernel_shape = self.kernel_size + (
            input_channel // self.groups,
            self.filters,
        )

        self.kernel_var_initializer = tf.keras.initializers.Constant(
                value=self.kernel_var_init)
        
        self.maxwent_kernel = self.add_weight(
            name="maxwent_kernel",
            shape=kernel_shape,
            initializer=self.kernel_var_initializer,
            dtype=self.dtype,
            trainable=True,
        )

        if self.use_bias:
            self.bias_var_initializer = tf.keras.initializers.Constant(
                value=self.bias_var_init)
            
            self.maxwent_bias = self.add_weight(
                name="maxwent_bias",
                shape=(self.filters,),
                initializer=self.bias_var_initializer,
                dtype=self.dtype,
                trainable=True,
            )
        
        self.dim_vmatrix_ = tf.reduce_prod(kernel_shape[:-1]).numpy()
        self.maxwent_Vmatrix = self.add_weight(
            name="maxwent_Vmatrix",
            shape=[self.dim_vmatrix_, self.dim_vmatrix_],
            initializer="zeros",
            dtype=self.dtype,
            trainable=False,
        )
        self.maxwent_Vmatrix.assign(tf.eye(self.dim_vmatrix_))

    def _z_sample(self, kind, shape):
        """
        Generates a random noise sample based on the specified distribution type.

        Args:
            kind (str): The type of noise distribution ('normal', 'uniform', 'bernoulli').
            shape (tuple): The shape of the output noise tensor.

        Returns:
            tf.Tensor: A tensor containing sampled noise values.

        Raises:
            ValueError: If an unknown noise distribution is provided.
        """
        if self.seed_ is None:
            rng = tf.random
        else:
            rng = tf.random.Generator.from_seed(self.seed_)
        
        if kind == "normal":
            z = rng.normal(shape)
        elif kind == "uniform":
            z = rng.uniform(shape) * 2. - 1.
        elif kind == "bernoulli":
            z = rng.uniform(shape)
            z = tf.cast(tf.math.greater(0.5, z), self.kernel.dtype) * 2. - 1.
        else:
            raise ValueError("Unknown noise distribution")
        return z

    def call(self, inputs):
        """
        Performs the forward pass, applying noise to kernel and bias.

        Args:
            inputs (tf.Tensor): Input tensor to the layer.

        Returns:
            tf.Tensor: The processed output tensor.
        """
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
                z = tf.reshape(z, (self.dim_vmatrix_, self.filters))
                z = tf.matmul(self.maxwent_Vmatrix, z)
                z = tf.reshape(z, self.kernel.shape)
            kernel = self.kernel + z
    
            if self.bias is not None:
                z = self._z_sample(self.bias_distrib, self.maxwent_bias.shape)
                z = tf.math.softplus(self.maxwent_bias) * z
                if self.clip_ is not None:
                    z = tf.clip_by_value(z, -self.clip_, self.clip_)
                bias = self.bias + z
        
        outputs = self.convolution_op(inputs, kernel)

        if self.use_bias:
            if self.data_format == "channels_last":
                bias_shape = (1,) * (self.rank + 1) + (self.filters,)
            else:
                bias_shape = (1, self.filters) + (1,) * self.rank
            bias = tf.reshape(self.bias, bias_shape)
            outputs = tf.add(outputs, bias)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def fit_svd(self, inputs, mode="train"):
        """
        Fit the Singular Value Decomposition (SVD) transition matrix on the input data.

        Args:
            inputs (tf.Tensor): Input data tensor.
            mode (str): The phase of SVD fitting ('start', 'train', 'end').
        """
        pass

class Conv1DMaxWEnt(BaseConvMaxWEnt):
    def __init__(
            self,
            kernel_distrib="uniform",
            bias_distrib="uniform",
            kernel_var_init=-7.,
            bias_var_init=-7.,
            **kwargs,
        ):
        super().__init__(rank=1, **kwargs)
        self.kernel_distrib = kernel_distrib
        self.bias_distrib = bias_distrib
        self.kernel_var_init = kernel_var_init
        self.bias_var_init = bias_var_init
        self.use_svd_ = False
        self.fit_svd_ = None
        self.clip_ = None
        self.seed_ = None
    
    def fit_svd(self, inputs, mode="train"):        
        if mode == "start":
            self.XTX_ = tf.zeros(self.maxwent_Vmatrix.shape)
            self.n_ = 0
            self.fit_svd_ = "train"
        
        elif mode == "train":
            if self.data_format != "channels_last":
                perm = [i for i in range(inputs.ndim) if i != 1] + [1]
                inputs = tf.transpose(inputs, perm=perm)
    
            input_channel = inputs.shape[-1]
    
            patching_kernel = tf.eye(self.kernel_size[0] * input_channel)
            patching_kernel = tf.reshape(patching_kernel,
                                         [self.kernel_size[0], input_channel,
                                         self.kernel_size[0] * input_channel])
            
            patches = tf.nn.conv1d(
                inputs,
                filters=patching_kernel,
                stride=self.strides[0],
                padding=self.padding.upper()
            )
    
            patches = tf.reshape(patches, (-1, patches.shape[-1]))
            print(patches.shape)
            
            self.XTX_ += tf.matmul(tf.transpose(patches), patches)
            self.n_ += patches.shape[0]
        
        elif mode == "end":
            self.XTX_ /= tf.cast(self.n_, self.XTX_.dtype)
            _, V = tf.linalg.eig(self.XTX_)
            V = tf.math.real(V)
            V = tf.cast(V, dtype=self.maxwent_Vmatrix.dtype)
            self.maxwent_Vmatrix.assign(V)
            self.fit_svd_ = None
            self.use_svd_ = True
            delattr(self, "XTX_")
            delattr(self, "n_")
        
        else:
            raise ValueError("mode should be in [start, train, end]")

class Conv2DMaxWEnt(BaseConvMaxWEnt):

    def __init__(
            self,
            kernel_distrib="uniform",
            bias_distrib="uniform",
            kernel_var_init=-7.,
            bias_var_init=-7.,
            **kwargs,
        ):
        super().__init__(rank=2, **kwargs)

        self.kernel_distrib = kernel_distrib
        self.bias_distrib = bias_distrib 
        self.kernel_var_init = kernel_var_init
        self.bias_var_init = bias_var_init
        self.use_svd_ = False
        self.fit_svd_ = None
        self.clip_ = None
        self.seed_ = None
    
    def fit_svd(self, inputs, mode="train"):
        if mode == "start":
            self.XTX_ = tf.zeros(self.maxwent_Vmatrix.shape)
            self.n_ = 0
            self.fit_svd_ = "train"
        
        elif mode == "train":

            if self.data_format != "channels_last":
                perm = [i for i in range(inputs.ndim) if i != 1] + [1]
                inputs = tf.transpose(inputs, perm=perm)
    
            input_channel = inputs.shape[-1]
    
            patching_kernel = tf.eye(self.kernel_size[0] *
                                     self.kernel_size[1] *
                                     input_channel)
            patching_kernel = tf.reshape(patching_kernel,
                                         [self.kernel_size[0],
                                          self.kernel_size[1],
                                          input_channel,
                                          patching_kernel.shape[0]])
            
            patches = tf.nn.conv2d(
                inputs,
                filters=patching_kernel,
                strides=self.strides,
                padding=self.padding.upper()
            )
    
            patches = tf.reshape(patches, (-1, patches.shape[-1]))
            print(patches.shape)
            
            self.XTX_ += tf.matmul(tf.transpose(patches), patches)
            self.n_ += patches.shape[0]
        
        elif mode == "end":
            self.XTX_ /= tf.cast(self.n_, self.XTX_.dtype)
            _, V = tf.linalg.eig(self.XTX_)
            V = tf.math.real(V)
            V = tf.cast(V, dtype=self.maxwent_Vmatrix.dtype)
            self.maxwent_Vmatrix.assign(V)
            self.fit_svd_ = None
            self.use_svd_ = True
            delattr(self, "XTX_")
            delattr(self, "n_")
        
        else:
            raise ValueError("mode should be in [start, train, end]")


class Conv3DMaxWEnt(BaseConvMaxWEnt):

    def __init__(
            self,
            kernel_distrib="uniform",
            bias_distrib="uniform",
            kernel_var_init=-7.,
            bias_var_init=-7.,
            **kwargs,
        ):
        super().__init__(rank=3, **kwargs)

        self.kernel_distrib = kernel_distrib
        self.bias_distrib = bias_distrib 
        self.kernel_var_init = kernel_var_init
        self.bias_var_init = bias_var_init
        self.use_svd_ = False
        self.fit_svd_ = None
        self.clip_ = None
        self.seed_ = None
    
    def fit_svd(self, inputs, mode="train"):
        if mode == "start":
            self.XTX_ = tf.zeros(self.maxwent_Vmatrix.shape)
            self.n_ = 0
            self.fit_svd_ = "train"
        
        elif mode == "train":

            if self.data_format != "channels_last":
                perm = [i for i in range(inputs.ndim) if i != 1] + [1]
                inputs = tf.transpose(inputs, perm=perm)
    
            input_channel = inputs.shape[-1]
    
            patching_kernel = tf.eye(self.kernel_size[0] *
                                     self.kernel_size[1] *
                                     self.kernel_size[2] *
                                     input_channel)
            patching_kernel = tf.reshape(patching_kernel,
                                         [self.kernel_size[0],
                                          self.kernel_size[1],
                                          self.kernel_size[2],
                                          input_channel,
                                          patching_kernel.shape[0]])
            
            patches = tf.nn.conv3d(
                inputs,
                filters=patching_kernel,
                strides=self.strides,
                padding=self.padding.upper()
            )
    
            patches = tf.reshape(patches, (-1, patches.shape[-1]))
            print(patches.shape)
            
            self.XTX_ += tf.matmul(tf.transpose(patches), patches)
            self.n_ += patches.shape[0]
        
        elif mode == "end":
            self.XTX_ /= tf.cast(self.n_, self.XTX_.dtype)
            _, V = tf.linalg.eig(self.XTX_)
            V = tf.math.real(V)
            V = tf.cast(V, dtype=self.maxwent_Vmatrix.dtype)
            self.maxwent_Vmatrix.assign(V)
            self.fit_svd_ = None
            self.use_svd_ = True
            delattr(self, "XTX_")
            delattr(self, "n_")
        
        else:
            raise ValueError("mode should be in [start, train, end]")