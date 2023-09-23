"""
This code contains the main network architectures used in the paper.
"""

import torch
import torch.nn as nn

from cifnet.dissimilarities import l_p


############################################################
#                                                          #
#                  Embedding Network                       #
#                                                          #
############################################################

class EmbeddingNet(nn.Module):
    """
    Embedding network for computing the embedding of the input data.
    Inputs:
        input_dim: int, dimensionality of the input data
        hidden_dim: int, dimensionality of the hidden layers
        embedding_dim: int, dimensionality of the embedding
    Outputs:
        output: torch.Tensor, embedding of the input
    """

    def __init__(self, input_dim, hidden_dim, embedding_dim):
        super(EmbeddingNet, self).__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.PReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, x):
        output = self.embedding(x)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class DeepEmbeddingNet(nn.Module):
    """
    Embedding network for computing the embedding of the input data.
    Inputs:
        input_dim: int, dimensionality of the input data
        hidden_dim: int, dimensionality of the hidden layers
        embedding_dim: int, dimensionality of the embedding
    Outputs:
        output: torch.Tensor, embedding of the input
    """

    def __init__(self, input_dim, hidden_dim, embedding_dim):
        super(DeepEmbeddingNet, self).__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.PReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, x):
        output = self.embedding(x)
        return output

    def get_embedding(self, x):
        return self.forward(x)

############################################################
#                                                          #
#                    Siamese Network                       #
#                                                          #
############################################################


class SiameseNet(nn.Module):
    """
    Siamese network for computing the embeddings of the two input samples and the distance between the embeddings.
    Inputs:
        embedding_net: nn.Module, embedding network
    """

    def __init__(self, embedding_net, margin, Q_metric=l_p):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net
        self.margin = margin
        self.Q_metric = Q_metric
        self.name = "Siamese"

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)

    def predict(self, test_dict):
        output1, output2 = self.forward(test_dict['X1'], test_dict['X2'])
        distance = self.Q_metric(output1, output2)
        label = (distance <= self.margin).to(int)
        return label, distance


############################################################
#                                                          #
#                    Triplet Network                       #
#                                                          #
############################################################


class TripletNet(nn.Module):
    """
    Triplet network for computing the embeddings of the three input samples and the distance between the embeddings.
    Inputs:
        embedding_net: nn.Module, embedding network
    """

    def __init__(self, embedding_net, margin, Q_metric=l_p):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net
        self.margin = margin
        self.Q_metric = Q_metric
        self.name = "Triplet"

    def forward(self, x_a, x_pos, x_neg):
        anchor = self.embedding_net(x_a)
        positive = self.embedding_net(x_pos)
        negative = self.embedding_net(x_neg)
        return anchor, positive, negative

    def get_embedding(self, x):
        return self.embedding_net(x)

    def predict(self, test_dict):
        anchor, positive, negative = self.forward(test_dict['X_a'], test_dict['X_pos'], test_dict['X_neg'])
        distance_positive = self.Q_metric(anchor, positive)
        distance_negative = self.Q_metric(anchor, negative)
        label = (distance_positive + self.margin < distance_negative).to(torch.int)
        return distance_positive, distance_negative, label

############################################################
#                                                          #
#         Siamese Causal Individual Fair Network           #
#                                                          #
############################################################

class CIFNet2(nn.Module):
    """
    Siamese causal individual fair network for computing the embeddings of the two input samples and the distance between the embeddings.
    Inputs:
        embedding_net: nn.Module, embedding network
    """

    def __init__(self, embedding_net):
        super(CIFNet2, self).__init__()
        self.embedding_net = embedding_net
        self.name = "CIFNet2"

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)


############################################################
#                                                          #
#         Triplet Causal Individual Fair Network           #
#                                                          #
############################################################

class CIFNet3(nn.Module):
    """
    Triplet causal individual fair network for computing the embeddings of the three input samples and the distance between the embeddings.
    Inputs:
        embedding_net: nn.Module, embedding network
    """

    def __init__(self, embedding_net):
        super(CIFNet3, self).__init__()
        self.embedding_net = embedding_net
        self.name = "CIFNet3"

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)
