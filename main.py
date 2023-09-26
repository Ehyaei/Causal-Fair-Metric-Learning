import os
from datetime import datetime

import numpy as np

from cifnet import utils
from cifnet.data_utils import learning_data, get_scm
from cifnet.networks import EmbeddingNet, SiameseNet, TripletNet, DeepEmbeddingNet
from cifnet.trainers import Trainer
from cifnet.utils import get_metric, result_to_DF


def run_benchmark(seed):
    """
    Run the benchmarking experiments.
    inputs:
        --seed: the random seed to use for the experiment
    outputs:
        train embedding mapping and metric


    data_name includes:
        - compas
        - adult
        - lin
        - nlm
        - imf
        - loan

    data_type includes:
        - contrastive
        - triplet

    decorrelation_loss_fn includes:
        - pearson
        - xicor
        - None

    output_type includes:
        - label
        - embedding

    metric_name includes:
        - l1
        - l2
        - l05
        - linf

    radi includes:
        - 0.1
        - 0.5
        - 1.0
    """

    data_names = ["compas", "adult", "lin", "nlm", "imf", "loan"]
    data_types = ["contrastive", "triplet"]
    decorrelation_loss_fns = ["pearson", "xicor", None]
    output_types = ["label", "embedding"]
    metric_names = ["l1", "l2", "l05", "linf"]
    radiis = [0.05, 0.1, 0.2]
    num_points = 10000
    epochs = 100
    hidden_dim = 50
    batch_size = 1000
    lambda_reg = 0.1
    device = 'cpu'
    counter = 0
    verbose = False

    dirs_2_create = [utils.model_save_dir, utils.metrics_save_dir, utils.scms_save_dir]
    for directory in dirs_2_create:
        if not os.path.exists(directory):
            os.makedirs(directory)

    for data_name in data_names:

        # get the SCM
        scm = get_scm(data_name)
        input_dim = len(scm.mean)
        embedding_dim = len(scm.mean) - len(scm.get_sensitive())
        embedding_net = DeepEmbeddingNet(input_dim, hidden_dim, embedding_dim)
        # embedding_net = EmbeddingNet(input_dim, hidden_dim, embedding_dim)

        for data_type in data_types:
            for metric_name in metric_names:

                # get the metric
                Q_metric = get_metric(metric_name)

                for radii in radiis:
                    # set model
                    data_dict = learning_data(data_name, data_type, num_points, radii, Q_metric)

                    if data_type == 'contrastive':
                        model = SiameseNet(embedding_net, radii, Q_metric)
                    elif data_type == 'triplet':
                        output_types = ["label"]
                        model = TripletNet(embedding_net, radii, Q_metric)

                    for decorrelation_loss_fn in decorrelation_loss_fns:
                        for output_type in output_types:
                            counter += 1

                            if data_type == 'contrastive':
                                margin = radii
                            else:
                                margin = 0.0

                            print("%d: data_name: %s, data_type: %s, decorrelation_loss_fn: "
                                  "%s, output_type: %s, Q_metric: %s, radi: %s" % (counter, data_name, data_type,
                                                                                   decorrelation_loss_fn,
                                                                                   output_type, metric_name, radii))

                            # Train the model
                            embedding_trainer = Trainer(batch_size=batch_size, lambda_reg=lambda_reg,
                                                        device=device, verbose=verbose)

                            acc, rmse, mae, mcc = embedding_trainer.train(model, data_dict, data_type, output_type,
                                                                          epochs, decorrelation_loss_fn, margin,
                                                                          Q_metric)

                            # Save the results
                            metrics_save_dir = utils.get_metrics_save_dir(data_name, data_type,
                                                                          decorrelation_loss_fn, output_type,
                                                                          metric_name, seed, radii, lambda_reg)
                            np.save(metrics_save_dir + '_acc.npy', np.array([acc]))
                            np.save(metrics_save_dir + '_rmse.npy', np.array([rmse]))
                            np.save(metrics_save_dir + '_mae.npy', np.array([mae]))
                            np.save(metrics_save_dir + '_mcc.npy', np.array([mcc]))

                            current_time = datetime.now()
                            formatted_time = current_time.strftime("%I:%M:%S %p")
                            # print("Is done at: ", formatted_time)
                            print(f"Acc: {acc:.4f} mcc: {mcc:.4f} rmse: {rmse:.4f} mae: {mae:.4f}")
                            # print("=============================================================")
    result_to_DF()


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    run_benchmark(args.seed)
