import numpy as np
from matplotlib.patches import Ellipse

def lighten_the_panes(ax):
    ax.xaxis.set_pane_color((1., 1., 1., 0.2))
    ax.yaxis.set_pane_color((1., 1., 1., 0.2))
    ax.zaxis.set_pane_color((1., 1., 1., 0.2))

def render_trajectories(coordinates, colors, min_val, max_val, ax):
    dim = coordinates[0].shape[-1]
    for trajectory, color in zip(coordinates, colors):
        trajectory = np.array(trajectory)
        if dim == 2:
            ax.plot(trajectory[:, 0], trajectory[:, 1], color = np.array(color[0]), linewidth=0.7)
        elif dim == 3:
            lighten_the_panes(ax)
            ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], color = np.array(color[0]), linewidth=0.7)
    
    offset = (max_val - min_val) * 0.1
    lim_min, lim_max = min_val - offset, max_val + offset
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    if dim == 3:
        ax.set_zlim(lim_min, lim_max)

def draw_2d_gaussians(means, stddevs, colors, ax, fill=False, alpha=0.8, use_adaptive_axis=False, draw_unit_gaussian=True, plot_axis=None):
    means = np.clip(means, -1000, 1000)
    stddevs = np.clip(stddevs, -1000, 1000)
    square_axis_limit = 2.0
    if draw_unit_gaussian:
        ellipse = Ellipse(xy=(0, 0), width=2, height=2,
                          edgecolor='r', lw=1, facecolor='none', alpha=0.5)
        ax.add_patch(ellipse)
    for mean, stddev, color in zip(means, stddevs, colors):
        if len(mean) == 1:
            mean = np.concatenate([mean, [0.]])
            stddev = np.concatenate([stddev, [0.1]])
        ellipse = Ellipse(xy=mean, width=stddev[0] * 2, height=stddev[1] * 2,
                          edgecolor=color, lw=1, facecolor='none' if not fill else color, alpha=alpha)
        ax.add_patch(ellipse)
        square_axis_limit = max(
                square_axis_limit,
                np.abs(mean[0] + stddev[0]),
                np.abs(mean[0] - stddev[0]),
                np.abs(mean[1] + stddev[1]),
                np.abs(mean[1] - stddev[1]),
        )
    square_axis_limit = square_axis_limit * 1.2
    ax.axis('scaled')
    if plot_axis is None:
        if use_adaptive_axis:
            ax.set_xlim(-square_axis_limit, square_axis_limit)
            ax.set_ylim(-square_axis_limit, square_axis_limit)
        else:
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
    else:
        ax.axis(plot_axis)

def calc_eval_metrics(coordinates, discretize_continuous_fn, prefix = None):
    eval_metrics = {}
    coordinate_seq = np.concatenate(coordinates, axis = 0)
    uniq_coords = np.unique(discretize_continuous_fn(coordinate_seq).reshape(-1, coordinate_seq.shape[-1]).astype(np.int32), axis=0)
    key = 'NumUniqueCoords'
    if prefix is not None:
        key = f'{prefix}_{key}'
    eval_metrics.update({
        key: len(uniq_coords),
    })
    return eval_metrics