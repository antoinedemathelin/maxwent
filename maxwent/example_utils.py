import numpy as np


def regression_1d(state=123, n_samples=100, noise=0.02):
    
    np.random.seed(state)
    def f(x):
        return 0.3 * x + 0.3 * np.sin(2*np.pi*x) + 0.3 * np.sin(4*np.pi*x)

    X_grid = np.linspace(-1.5, 1.5, 1000)
    y_grid = f(X_grid)

    n = int(n_samples / 2)
    x = np.random.randn(n)*0.1 - 0.5
    x = np.concatenate((x, np.random.randn(n)*0.1 + 0.75))
    y = f(x) + noise * np.random.randn(x.shape[0])

    return x.reshape(-1, 1), y, X_grid.reshape(-1, 1), y_grid


def classification_2d(state=0, n_samples=200, noise=0.1):
    
    np.random.seed(state)
    try:
        from sklearn.datasets import make_moons
        x, y = make_moons(n_samples=n_samples, noise=noise)
    except:
        raise ImportError("The scikit-learn library is not installed."
                            " The classification_2d function requires the"
                            " scikit-learn library.")
    x = x - np.array([0.5, 0.25])
    
    x_min, y_min = -3., -3
    x_max, y_max = 3, 3
    x_grid, y_grid = np.meshgrid(np.linspace(x_min-0.1, x_max+0.1, 100),
                                 np.linspace(y_min-0.1, y_max+0.1, 100))
    X_grid = np.stack([x_grid.ravel(), y_grid.ravel()], -1)
    
    return x, y, X_grid


def plot_regression_1d(x_train, y_train, x_ood, y_ood):

    try:
        import matplotlib.pyplot as plt
    except:
        raise ImportError("The matplotlib library is not installed."
                            " The plot_regression_1d function requires the"
                            " matplotlib library.")

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        
    ax.plot(x_ood.ravel(), y_ood, c="k", label="Ground truth")
    ax.plot(x_train, y_train, "o", c="w", markeredgecolor="k", markersize=5, label="Training data")
    ax.set_xlabel(r"$\mathcal{X}$", fontsize=20)
    ax.set_ylabel(r"$\mathcal{Y}$", fontsize=20, rotation=0, labelpad=15)
    ax.tick_params(direction = "in", left = True, right = False , labelleft = False ,
                    labelbottom = False, bottom = True, top = False)
    ax.set_ylim(-2, 2.); ax.set_xlim(-1.48, 1.48)
    return ax


def plot_classification_2d(x_train, y_train, x_ood):

    try:
        import matplotlib.pyplot as plt
    except:
        raise ImportError("The matplotlib library is not installed."
                            " The plot_classification_2d function requires the"
                            " matplotlib library.")

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        
    ax.plot(x_train[y_train==0, 0], x_train[y_train==0, 1], 'o',
            c="w", markeredgecolor="k", markersize=4,
            label="Training data (Class 0)")
    ax.plot(x_train[y_train==1, 0], x_train[y_train==1, 1], 's',
            c="w", markeredgecolor="k", markersize=4,
            label="Training data (Class 1)")
    ax.set_xlabel(r"$\mathcal{X}_1$", fontsize=20)
    ax.set_ylabel(r"$\mathcal{X}_2$", fontsize=20, rotation=0, labelpad=15)
    ax.tick_params(direction = "in", left = True, right = False , labelleft = False ,
                    labelbottom = False, bottom = True, top = False)
    ax.set_xlim(x_ood[:, 0].min(), x_ood[:, 0].max())
    ax.set_ylim(x_ood[:, 1].min(), x_ood[:, 1].max())
    return ax