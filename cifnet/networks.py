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

    def __init__(self, embedding_net, radii, Q_metric=l_p):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net
        self.radii = radii
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
        label = (distance <= self.radii).to(int)
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

    def __init__(self, embedding_net, radii, Q_metric=l_p):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net
        self.radii = radii
        self.Q_metric = Q_metric
        self.name = "Triplet"

    def forward(self, x_a, x_1, x_2):
        anchor = self.embedding_net(x_a)
        positive = self.embedding_net(x_1)
        negative = self.embedding_net(x_2)
        return anchor, positive, negative

    def get_embedding(self, x):
        return self.embedding_net(x)

    def predict(self, test_dict):
        anchor, X1, X2 = self.forward(test_dict['X_a'], test_dict['X_1'], test_dict['X_2'])
        d1 = self.Q_metric(anchor, X1)
        d2 = self.Q_metric(anchor, X2)
        label = (d1 < d2).to(torch.int)
        return d1, d2, label
