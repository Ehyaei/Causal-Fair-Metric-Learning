"""
This code contains the loss functions for metric learning.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from cifnet.dissimilarities import l_p
from cifnet.xicor import xicorrcoef
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


############################################################
#                                                          #
#                        Huber Loss Function               #
#                                                          #
############################################################

class HuberLoss(nn.Module):
    """
    Implementation of Huber loss function.
    """

    def __init__(self, Q_metric):
        super(HuberLoss, self).__init__()
        self.eps = 1e-9
        self.Q_metric = Q_metric

    def forward(self, output1, output2, target, size_average=False):
        distances = self.Q_metric(output1, output2)
        losses = F.huber_loss(target.float(), distances)

        return losses.mean() if size_average else losses.sum()


############################################################
#                                                          #
#                  Contrastive Loss Function               #
#                                                          #
############################################################

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin, Q_metric):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.Q_metric = Q_metric

    def forward(self, output1, output2, target, size_average=False):
        distance = self.Q_metric(output1, output2)
        losses = target.float() * distance + (1 + -1 * target).float() * F.relu(self.margin - distance)
        return losses.mean() if size_average else losses.sum()


############################################################
#                                                          #
#                    Triplet Loss Function                 #
#                                                          #
############################################################


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin, Q_metric):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.Q_metric = Q_metric

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = self.Q_metric(anchor, positive)
        distance_negative = self.Q_metric(anchor, negative)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


############################################################
#                                                          #
#                  Quadruple Loss Function                 #
#                                                          #
############################################################


class QuadrupletLoss(nn.Module):
    """
    Quadruplet loss
    Takes embeddings of an anchor sample, a positive sample, and two negative samples
    """

    def __init__(self, margin, Q_metric):
        super(QuadrupletLoss, self).__init__()
        self.margin = margin
        self.Q_metric = Q_metric

    def forward(self, anchor, positive, negative1, negative2, size_average=True):
        distance_positive = self.Q_metric(anchor, positive)
        distance_negative1 = self.Q_metric(anchor, negative1)
        distance_negative2 = self.Q_metric(anchor, negative2)

        # Calculate the loss based on the margin and differences between distances
        losses = F.relu(distance_positive - distance_negative1 + self.margin) + F.relu(
            distance_positive - distance_negative2 + self.margin)

        return losses.mean() if size_average else losses.sum()


############################################################
#                                                          #
#                  Angular Loss Function                   #
#                                                          #
############################################################
# TODO: Implement Angular Loss Function


############################################################
#                                                          #
#              Choose the Loss w.r.t Data                  #
#                                                          #
############################################################
def loss_fn(model, batch, data_type, output_type, margin=10.0, Q_metric=l_p):
    if data_type == 'contrastive':
        X1, X2, label, distance = batch
        O1, O2 = model(X1, X2)
        if output_type == 'label':
            loss = ContrastiveLoss(margin, Q_metric)(O1, O2, label)
        elif output_type == 'embedding':
            loss = HuberLoss(Q_metric)(O1, O2, distance)
        else:
            raise AssertionError
    elif data_type == 'triplet':
        X_a, X_1, X_2, _, _, _ = batch
        O_a, O_pos, O_neg = model(X_a, X_1, X_2)
        loss = TripletLoss(margin, Q_metric)(O_a, O_pos, O_neg)
    else:
        raise AssertionError
    return loss


############################################################
#                                                          #
#              Decorrelation Loss Function                 #
#                                                          #
############################################################


class DecorrelationLoss(nn.Module):
    """
    This loss function is used to decorrelate the features of the embedding network based on the
    pearson correlation coefficient.
    inputs:
        method: str, method to compute the correlation matrix, either 'pearson' or 'xicor'
    """

    def __init__(self, method='pearson'):
        super(DecorrelationLoss, self).__init__()
        self.type = method

    def forward(self, x):
        # Compute the covariance matrix of the input tensor
        if self.type == 'pearson':
            # Compute the pearson correlation matrix
            cov = torch.corrcoef(x.T)
        else:
            # Compute the Xi correlation matrix
            cov = xicorrcoef(x.detach().numpy())

        # Remove the diagonal and compute the decorrelation loss as the Frobenius norm of the covariance matrix
        cov = torch.tensor(cov)
        loss = torch.norm(cov - torch.eye(x.shape[1], device=x.device), p='fro')

        return loss


def dec_loss_fn(model, batch, data_type, method):
    """
    This function computes the decorrelation loss for the given batch.
    :param model: Input model network
    :param batch: Input batch
    :param data_type: Input data type, either contrastive or triplet
    :param method: Input method to compute the correlation matrix, either 'pearson' or 'xicor'
    :return: decorrelation loss
    """
    if data_type == 'contrastive':
        X1, X2, label, distance = batch
        Output = torch.cat(model(X1, X2), dim=0)
    elif data_type == 'triplet':
        X_a, X_1, X_2, _, _, _ = batch
        Output = torch.cat(model(X_a, X_1, X_2), dim=0)
    else:
        raise AssertionError

    loss = DecorrelationLoss(method)(Output)

    return loss

############################################################
#                                                          #
#                 Likelihood Loss Function                 #
#                                                          #
############################################################

# TODO: Implement Likelihood Loss Function
