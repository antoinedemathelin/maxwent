import tensorflow as tf
from maxwent import MaxWEnt, set_maxwent_model

def test_minimal():
    X = tf.ones((10, 2))
    y = tf.ones((10,))
    base_net = tf.keras.Sequential()
    base_net.add(tf.keras.layers.Dense(10, activation="relu"))
    base_net.add(tf.keras.layers.Dense(1))
    base_net.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(0.001))
    base_net.fit(X, y, epochs=1)

    net = set_maxwent_model(base_net)
    mwe = MaxWEnt(net)
    mwe.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(0.001))
    mwe.fit(X, y, epochs=1)
    
    net = set_maxwent_model(base_net)
    mwe = MaxWEnt(net)
    mwe.fit_svd(X)
    mwe.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(0.001))
    mwe.fit(X, y, epochs=1)
    