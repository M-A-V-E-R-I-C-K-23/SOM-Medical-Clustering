"""
som_model.py -- SOM training and analysis utilities for Medical SOM project.

y (diagnosis labels) NEVER enters this module. All training is on X_scaled only.
"""

import numpy as np
import matplotlib.pyplot as plt
from minisom import MiniSom

DEFAULT_GRID  = (11, 11)  # Phase 3 baseline prototype; final selected grid is 9x9
DEFAULT_SIGMA = 1.5
DEFAULT_LR    = 0.5
DEFAULT_ITERS = 10000
DEFAULT_SEED  = 42


def create_som(n_rows=DEFAULT_GRID[0], n_cols=DEFAULT_GRID[1], n_features=30,
               sigma=DEFAULT_SIGMA, learning_rate=DEFAULT_LR,
               random_seed=DEFAULT_SEED, topology='rectangular'):
    """Instantiate an untrained MiniSom grid."""
    return MiniSom(x=n_rows, y=n_cols, input_len=n_features, sigma=sigma,
                   learning_rate=learning_rate, random_seed=random_seed,
                   topology=topology)


def train_som(som, X_scaled, n_iterations=DEFAULT_ITERS,
              random_seed=DEFAULT_SEED, record_qe_every=500):
    """Initialize weights and train the SOM on X_scaled. Returns (som, qe_history)."""
    np.random.seed(random_seed)
    som.random_weights_init(X_scaled)

    qe_history = []
    step_indices = list(range(0, n_iterations + 1, record_qe_every))
    if step_indices[-1] != n_iterations:
        step_indices.append(n_iterations)

    n_samples = len(X_scaled)
    rng = np.random.default_rng(random_seed)

    print(f'Training SOM {som._weights.shape[0]}x{som._weights.shape[1]} '
          f'for {n_iterations} iterations ...')

    trained_steps = 0
    for target_step in step_indices:
        steps_to_run = target_step - trained_steps
        if steps_to_run <= 0:
            qe_history.append((trained_steps, som.quantization_error(X_scaled)))
            continue
        for _ in range(steps_to_run):
            idx = rng.integers(0, n_samples)
            som.update(X_scaled[idx], som.winner(X_scaled[idx]), trained_steps, n_iterations)
            trained_steps += 1
        qe = som.quantization_error(X_scaled)
        qe_history.append((trained_steps, qe))
        print(f'  step {trained_steps:6d}/{n_iterations}  |  QE = {qe:.5f}')

    print('Training complete.')
    return som, qe_history


def get_bmu_indices(som, X_scaled):
    """Return (row, col) BMU for every sample. Shape: (n_samples, 2)."""
    return np.array([som.winner(x) for x in X_scaled])


def get_umatrix(som):
    """U-Matrix: average distance between each neuron and its neighbours."""
    return som.distance_map()


def get_hit_map(som, X_scaled):
    """Count of training samples mapped to each neuron. Shape: (n_rows, n_cols)."""
    n_rows, n_cols = som._weights.shape[:2]
    hit_map = np.zeros((n_rows, n_cols), dtype=int)
    bmus = np.array([som.winner(x) for x in X_scaled])
    np.add.at(hit_map, (bmus[:, 0], bmus[:, 1]), 1)
    return hit_map


def get_codebook(som):
    """Return the full neuron weight matrix. Shape: (n_rows, n_cols, n_features)."""
    return som.get_weights()


def get_som_summary(som, X_scaled, n_iterations, sigma, learning_rate, random_seed):
    """Return a dict of key SOM metrics and hyperparameters (used by notebook)."""
    n_rows, n_cols = som._weights.shape[:2]
    hit_map = get_hit_map(som, X_scaled)
    bmu_idx = get_bmu_indices(som, X_scaled)
    return {
        'grid_size':           f'{n_rows}x{n_cols}',
        'n_neurons':           n_rows * n_cols,
        'n_features':          som._weights.shape[2],
        'n_training_samples':  len(X_scaled),
        'n_iterations':        n_iterations,
        'sigma':               sigma,
        'learning_rate':       learning_rate,
        'random_seed':         random_seed,
        'quantization_error':  round(som.quantization_error(X_scaled), 6),
        'topographic_error':   round(som.topographic_error(X_scaled), 6),
        'dead_neurons':        int((hit_map == 0).sum()),
        'max_hits_per_neuron': int(hit_map.max()),
        'unique_bmus':         len(set(map(tuple, bmu_idx))),
    }


def plot_umatrix(som, save_path=None, title='U-Matrix'):
    """Plot the U-Matrix. Returns fig."""
    umatrix = get_umatrix(som)
    n_rows, n_cols = umatrix.shape
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(umatrix.T, cmap='viridis_r', origin='upper',
                   interpolation='nearest', aspect='equal')
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label('Mean distance to neighbours', fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Neuron column', fontsize=11)
    ax.set_ylabel('Neuron row', fontsize=11)
    ax.set_xticks(range(n_cols)); ax.set_yticks(range(n_rows))
    ax.tick_params(labelsize=8)
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=0.4)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    return fig


def plot_hit_map(som, X_scaled, save_path=None, title='BMU Hit Map'):
    """Plot the hit map. Returns fig."""
    hit_map = get_hit_map(som, X_scaled)
    n_rows, n_cols = hit_map.shape
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(hit_map.T, cmap='YlOrRd', origin='upper',
                   interpolation='nearest', aspect='equal')
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label('Number of samples (hits)', fontsize=11)
    for r in range(n_rows):
        for c in range(n_cols):
            count = hit_map[r, c]
            color = 'white' if count > hit_map.max() * 0.6 else 'black'
            ax.text(c, r, str(count), ha='center', va='center',
                    fontsize=6, color=color, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Neuron column', fontsize=11)
    ax.set_ylabel('Neuron row', fontsize=11)
    ax.set_xticks(range(n_cols)); ax.set_yticks(range(n_rows))
    ax.tick_params(labelsize=8)
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=0.5)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    return fig


def plot_training_curve(qe_history, save_path=None,
                        title='SOM Training: Quantization Error vs Iterations'):
    """Plot QE convergence curve. Returns fig."""
    steps = [h[0] for h in qe_history]
    qes   = [h[1] for h in qe_history]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, qes, color='#3498db', linewidth=2.5, marker='o',
            markersize=5, markerfacecolor='white', markeredgewidth=2)
    ax.fill_between(steps, qes, alpha=0.12, color='#3498db')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Training Iterations', fontsize=12)
    ax.set_ylabel('Quantization Error', fontsize=12)
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.35)
    ax.annotate(f'Final QE = {qes[-1]:.4f}', xy=(steps[-1], qes[-1]),
                xytext=(-120, 20), textcoords='offset points', fontsize=11,
                color='#2c3e50', arrowprops=dict(arrowstyle='->', color='#2c3e50'))
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    return fig
