"""
Utility functions for the project.
"""
import os

import numpy as np
import pandas as pd
import torch

from cifnet.dissimilarities import l1, l2, l05, linf

model_save_dir = 'models/'
metrics_save_dir = 'results/'
scms_save_dir = 'scms/'
DATA_DIR = 'dataset/'


def get_data_file(data_name):
    return os.path.join(DATA_DIR, '%s.csv' % data_name)


def sample_ball(num_points, dimension, radius):
    """
    Generate N samples inside D-dimensional unit ball with radius Delta.
    Inputs:     n: int, number of samples
                d: int, dimensionality of the samples
                delta: float, radius of the ball
    Returns:    z: np.array (N, D), samples
    """
    num_points = int(num_points)
    dimension = int(dimension)
    z = np.random.randn(num_points, dimension)
    z = z / np.linalg.norm(z, axis=1)[:, None]
    r = np.random.rand(num_points) ** (1 / dimension)
    return radius * r[:, None] * z


def closest_points(center, radius, metric_func, num_points=1000):
    """
    Generate points within a unit ball defined by the given metric.

    Args:
        center (torch.Tensor): The center point of the unit ball.
        radius (float): The radius of the unit ball.
        metric_func: A function that computes the distance metric between two points.
        num_points (int): The number of points to generate within the unit ball.

    Returns:
        torch.Tensor: A tensor containing the generated points within the unit ball.
    """
    points = []
    distances = []

    while len(points) < num_points:
        # Generate a random point within a unit sphere in a high-dimensional space
        random_point = torch.randn_like(center)
        start_point = center + random_point
        d = metric_func(center, start_point)
        random_point /= d

        # Scale the random point by the radius and add it to the center
        random_number = torch.rand(1)
        scaled_point = center + radius * random_number * random_point
        d = metric_func(center, scaled_point)
        # Check if the scaled point is within the unit ball
        if d <= radius:
            points.append(scaled_point)
            distances.append(d)

    return torch.stack(points), torch.stack(distances)


def farthest_points(center, radius, metric_func, num_points=1000):
    """
    Generate points outside a unit ball defined by the given metric.

    Args:
        center (torch.Tensor): The center point of the unit ball.
        radius (float): The radius of the unit ball.
        metric_func: A function that computes the distance metric between two points.
        num_points (int): The number of points to generate within the unit ball.

    Returns:
        torch.Tensor: A tensor containing the generated points within the unit ball.
    """
    points = []
    distances = []
    while len(points) < num_points:
        # Generate a random point within a unit sphere in a high-dimensional space
        random_point = torch.randn_like(center)
        scaled_point = center + random_point
        d = metric_func(center, scaled_point)
        random_point /= d

        # Scale the random point by the radius and add it to the center
        random_number = 1 + 4 * torch.rand(1)
        scaled_point = center + radius * random_number * random_point
        d = metric_func(center, scaled_point)
        # Check if the scaled point is within the unit ball
        if d > radius:
            points.append(scaled_point)
            distances.append(d)

    return torch.stack(points), torch.stack(distances)


def random_sample(data, sample_num, replace=False):
    """
    Randomly sample a given number of points from a tensor.
    Inputs:
        data: np.array, the data to sample from
        sample_num: int, the number of points to sample
        replace: bool, whether to sample with replacement
    Returns:
        np.array, the sampled points
    """
    num_points = data.shape[0]
    indices = np.random.choice(num_points, sample_num, replace)
    return data[indices, :]


def get_metric(metric_name):
    if metric_name == "l1":
        return l1
    elif metric_name == "l2":
        return l2
    elif metric_name == "l05":
        return l05
    elif metric_name == "linf":
        return linf
    else:
        raise ValueError("Unknown metric name: %s" % metric_name)


def get_model_save_dir(data_name, data_type, decorrelation_loss_fn, output_type, metric_name, seed, margin, lambd):
    model_dir = model_save_dir + '%s_%s_%s_%s_%s_%.0f_%.2f_%.2f' % (
        data_name, data_type, decorrelation_loss_fn, output_type, metric_name, seed, margin, lambd)

    return model_dir


def get_metrics_save_dir(data_name, data_type, decorrelation_loss_fn, output_type, metric_name, seed, margin, lambd):
    return metrics_save_dir + '%s_%s_%s_%s_%s_%.0f_%.2f_%.2f' % (
        data_name, data_type, decorrelation_loss_fn, output_type, metric_name, seed, margin, lambd)


def result_to_DF(path='results/', save='plots/'):
    files = os.listdir(path)
    df_cols = ["data", "type", "decorrelation", "output_type", "metric", "seed", "radii", "lambda", "indicator"]
    split_columns = [s.replace('.npy', '').split("_") for s in files]
    df = pd.DataFrame(split_columns, columns=df_cols)
    df['value'] = np.nan
    for i in range(len(files)):
        df.loc[i, "value"] = float(np.load(path + files[i]))

    df.to_csv(save + 'results.csv', index=False)
    return "File saved to " + save + 'results.csv'
