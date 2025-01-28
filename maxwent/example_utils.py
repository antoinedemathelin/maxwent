import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons


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

    return (x.reshape(-1, 1), y, X_grid.reshape(-1, 1), y_grid)


def classification_2d(state=0, n_samples=200, noise=0.1):
    
    np.random.seed(state)
    x, y = make_moons(n_samples=n_samples, noise=noise)
    x = x - np.array([0.5, 0.25])
    
    x_min, y_min = -3., -3
    x_max, y_max = 3, 3
    x_grid, y_grid = np.meshgrid(np.linspace(x_min-0.1, x_max+0.1, 100),
                                 np.linspace(y_min-0.1, y_max+0.1, 100))
    X_grid = np.stack([x_grid.ravel(), y_grid.ravel()], -1)
    
    return x, y, X_grid, None


def plot_regression_1d(data, model=None, n_preds=50, plot_conf=False, plot_models=True):
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    x, y, X_grid, y_grid = data
    
    if model is not None:
        preds = []
        for _ in range(n_preds):
            yp = model(X_grid, training=False).numpy()
            preds.append(yp)
        preds = np.stack(preds, axis=-1)
        y_mu = preds.mean(-1).ravel()
        y_std = preds.std(-1).ravel()

        if plot_conf:
            ax.plot(X_grid.ravel(), y_mu, c="C0")
            ax.fill_between(X_grid.ravel(),
                             y_mu-2*y_std,
                             y_mu+2*y_std,
                             color="C0", alpha=0.3, label="Confidence Intervals")
            ax.plot(X_grid.ravel(), y_mu-2*y_std, c="C0", alpha=0.5)
            ax.plot(X_grid.ravel(), y_mu+2*y_std, c="C0", alpha=0.5)
        if plot_models:
            for i in range(n_preds):
                ax.plot(X_grid.ravel(), preds[:, :, i].ravel(), c="purple")
        
    ax.plot(X_grid.ravel(), y_grid, c="k", label="Ground Truth")
    ax.plot(x, y, "o", c="w", markeredgecolor="k", markersize=5, label="Training data")
    ax.set_xlabel(r"$\mathcal{X}$", fontsize=20)
    ax.set_ylabel(r"$\mathcal{Y}$", fontsize=20, rotation=0, labelpad=15)
    ax.tick_params(direction = "in", left = True, right = False , labelleft = False ,
                    labelbottom = False, bottom = True, top = False)
    ax.set_ylim(-2, 2.); ax.set_xlim(-1.48, 1.48)
    ax.legend(); plt.show()
    return ax