
"""
This Code contains the Trainer class for training the models.
"""
import torch
from torch.utils.data import TensorDataset

from cifnet.data_utils import dict_to_train_test
from cifnet.dissimilarities import l_p
from cifnet.losses import loss_fn, dec_loss_fn
from cifnet.metrics import performance_metrics


# ----------------------------------------------------------------------------------------------------------------------
#
#                                                         BASE CLASS
#
# ----------------------------------------------------------------------------------------------------------------------


class Trainer:
    def __init__(self, lr=0.001, batch_size=1000, lambda_reg=0, device='cpu', verbose=False,
                 print_freq=1, save_freq=None, save_dir=None):
        self.lr = lr
        self.batch_size = batch_size
        self.lambda_reg = lambda_reg
        self.device = device
        self.verbose = verbose
        self.print_freq = print_freq
        self.save_model = (save_freq is not None) and (save_dir is not None)
        self.save_freq = save_freq
        self.save_dir = save_dir

    def train(self, model, data_dict, data_type, output_type, epochs, decorrelation_loss_fn=None, margin=10.0,
              Q_metric=l_p):
        """
        This function trains the given model on the given data.
        :param model: input model network
        :param data_dict: training data dictionary
        :param data_type: type of the data, either contrastive or triplet
        :param output_type: type of the output, either label or embedding
        :param epochs: number of epochs to train
        :param decorrelation_loss_fn: type of the decorrelation loss function, either pearson or xicor of None
        :param margin: margin for triplet loss or diameter for distance loss
        :param Q_metric: semi-latent metric
        :return: model metrics
        """

        train_dict, test_dict = dict_to_train_test(data_dict, ratio=0.8, tensor=True, device=self.device)

        tensors = list(train_dict.values())
        train_dst = TensorDataset(*tensors)
        train_loader = torch.utils.data.DataLoader(dataset=train_dst, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        for epoch in range(epochs):
            for batch in train_loader:
                optimizer.zero_grad()

                loss = loss_fn(model, batch, data_type, output_type, margin, Q_metric)
                if decorrelation_loss_fn is not None:
                    dec_loss = dec_loss_fn(model, batch, data_type, decorrelation_loss_fn)
                    # print("Deccoralation Loss: ", round(self.lambda_reg *dec_loss.detach().item(), 4), "    Loss", round(loss.detach().item(), 4))
                    loss += self.lambda_reg * dec_loss

                loss.backward(retain_graph=True)
                optimizer.step()

            if (epoch % self.print_freq == 0) & self.verbose:
                acc, rmse, mae, mcc, fp, fn = performance_metrics(model, test_dict, data_type)
                print(f"{epoch}    Acc: {acc:.4f} mcc: {mcc:.4f} rmse: {rmse:.4f} mae: {mae:.4f}  fp: {fp:.4f} fn: {fn:.4f}")

            if self.save_model:
                if (epoch % self.save_freq) == 0 and epoch > 0:
                    torch.save(model.state_dict(), self.save_dir + '_e' + str(epoch) + '.pth')

        if self.save_model:
            torch.save(model.state_dict(), self.save_dir + '.pth')

        return performance_metrics(model, test_dict, data_type)
