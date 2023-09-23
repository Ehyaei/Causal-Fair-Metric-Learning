"""
This file contains the implementation of the Structural Causal Models used for modelling the effect of interventions
on the features of the individual seeking recourse. This code adopted from Adversarially Robust Recourse repository
https://github.com/RicardoDominguez/AdversariallyRobustRecourse/tree/main/src
"""

import numpy as np
import torch
from cifnet.utils import sample_ball, closest_points, random_sample, farthest_points
from cifnet.dissimilarities import l_p


class SCM:
    """
    Includes all the relevant methods required for generating counterfactuals. Classes inheriting this class must
    contain the following objects:
        self.f: list of functions, each representing a structural equation. Function self.f[i] must have i+1 arguments,
                corresponding to X_1, ..., X_{i-1}, U_{i+1} each being a torch.Tensor with shape (N, 1), and returns
                the endogenous variable X_{i+1} as a torch.Tensor with shape (N, 1).
        self.inv_f: list of functions, corresponding to the inverse mapping X -> U. Each function self.inv_f[i] takes
                    as argument the features X as a torch.Tensor with shape (N, D), and returns the corresponding.
                    exogenous variable U_{i+1} as a torch.Tensor with shape (N, 1).
        self.actionable: list of int, indices of the actionable features.
        self.sensitive: list of int, indices of the sensitive features.
        self.soft_interv: list of bool with len = D, indicating whether the intervention on feature soft_interv[i] is
                          modeled as a soft intervention (True) or hard intervention (False).
        self.mean: expectation of the features, such that when generating data we can standardize it.
        self.std: standard deviation of the features, such that when generating data we can standardize it.
    """

    def sample_U(self, N):
        """
        Return N samples from the distribution over exogenous variables P_U.

        Inputs:     N: int, number of samples to draw

        Outputs:    U: np.array with shape (N, D)
        """
        raise NotImplementedError

    def label(self, X):
        """
        Label the input instances X

        Inputs:     X: np.array with shape (N, D)

        Outputs:    Y:  np.array with shape (N, )
        """
        raise NotImplementedError

    def generate(self, N):
        """
        Sample from the observational distribution implied by the SCM

        Inputs:     N: int, number of instances to sample

        Outputs:    X: np.array with shape (N, D), standardized (since we train the models on standardized data)
                    Y: np.array with shape (N, )
        """
        U = self.sample_U(N).astype(np.float32)
        X = self.U2X(torch.Tensor(U))
        Y = self.label(X.detach().numpy())
        X = (X - self.mean) / self.std

        return X.detach().numpy(), Y

    def U2X(self, U):
        """
        Map from the exogenous variables U to the endogenous variables X by using the structural equations self.f

        Inputs:     U: torch.Tensor with shape (N, D), exogenous variables

        Outputs:    X: torch.Tensor with shape (N, D), endogenous variables
        """
        X = []
        for i in range(U.shape[1]):
            X.append(self.f[i](*X[:i] + [U[:, [i]]]))
        return torch.cat(X, 1)

    def X2U(self, X):
        """
        Map from the endogenous variables to the exogenous variables by using the inverse mapping self.inv_f

        Inputs:     U: torch.Tensor with shape (N, D), exogenous variables

        Outputs:    X: torch.Tensor with shape (N, D), endogenous variables
        """
        if self.inv_f is None:
            return X + 0.
        U = torch.zeros_like(X)
        for i in range(X.shape[1]):
            U[:, [i]] = self.inv_f[i](X)
        return U

    def X2Q(self, X, sensitive=None):
        """
        Map from the endogenous variables to the semi-latent space by using the inverse mapping self.inv_f

        Inputs:     U: torch.Tensor with shape (N, D), exogenous variables
                    sensitive: None or list of int, indices of the sensitive features

        Outputs:    Q: torch.Tensor with shape (N, D), endogenous variables
        """
        sensitive = self.sensitive if sensitive is None else sensitive
        U = self.X2U(X)
        Q = U.clone()
        for i in sensitive:
            Q[:, [i]] = X[:, [i]]

        return Q

    def Xn2Q(self, Xn, sensitive=None):
        """
        Map from the endogenous variables to the semi-latent space by using the inverse mapping self.inv_f

        Inputs:     Xn: torch.Tensor with shape (N, D), exogenous variables
                    sensitive: None or list of int, indices of the sensitive features

        Outputs:    Q: torch.Tensor with shape (N, D), semi-latent variables
        """
        sensitive = self.sensitive if sensitive is None else sensitive
        return self.X2Q(self.Xn2X(Xn), sensitive)

    def Q2X(self, Q, sensitive=None):
        """
        Map from the endogenous variables to the semi-latent space by using the inverse mapping self.inv_f

        Inputs:     Q: torch.Tensor with shape (N, D), exogenous variables
                    sensitive: None or list of int, indices of the sensitive features

        Outputs:    X: torch.Tensor with shape (N, D), endogenous variables
        """
        sensitive = self.sensitive if sensitive is None else sensitive
        X = []
        for i in range(Q.shape[1]):
            if i in sensitive:
                X.append(Q[:, [i]])
            else:
                X.append(self.f[i](*X[:i] + [Q[:, [i]]]))

        X = torch.cat(X, 1)
        return X

    def Q2Xn(self, Q, sensitive=None):
        """
        Map from the endogenous variables to the semi-latent space by using the inverse mapping self.inv_f

        Inputs:     U: torch.Tensor with shape (N, D), exogenous variables
                    sensitive: None or list of int, indices of the sensitive features

        Outputs:    Q: torch.Tensor with shape (N, D), endogenous variables
        """
        sensitive = self.sensitive if sensitive is None else sensitive
        return self.X2Xn(self.Q2X(Q, sensitive))

    def counterfactual(self, Xn, delta, actionable, soft_interv):
        """
        Computes the counterfactual of Xn under the intervention delta.

        Inputs:     Xn: torch.Tensor (N, D) factual
                    delta: torch.Tensor (N, D), intervention values
                    actionable: None or list of int, indices of the intervened upon variables
                    soft_interv: None or list of Boolean, variables for which the interventions are soft

        Outputs:
                    X_cf: torch.Tensor (N, D), counterfactual
        """
        actionable = self.actionable if actionable is None else actionable
        soft_interv = self.soft_interv if soft_interv is None else soft_interv

        # Abduction
        X = self.Xn2X(Xn)
        U = self.X2U(X)

        # Scale appropriately
        delta = delta * self.std

        X_cf = []
        for i in range(U.shape[1]):
            if i in actionable:
                if soft_interv[i]:
                    X_cf.append(self.f[i](*X_cf[:i] + [U[:, [i]] + delta[:, [i]]]))
                else:
                    X_cf.append(delta[:, [i]])
            else:
                X_cf.append(self.f[i](*X_cf[:i] + [U[:, [i]]]))

        X_cf = torch.cat(X_cf, 1)
        return self.X2Xn(X_cf)

    def sensitive_levels(self, sensitive=None, sample_n=10000):
        sensitive = self.sensitive if sensitive is None else sensitive

        # generate samples and find levels of sensitive features
        Sn, _ = self.generate(sample_n)

        # convert to tensor
        Sn = torch.from_numpy(Sn)

        # Set non-sensitive features to zero
        for i in range(Sn.shape[1]):
            if i not in sensitive:
                Sn[:, [i]] = 0.

        # Find Sensitive levels
        return torch.unique(Sn, dim=0)


    def twins(self, Xn, sensitive, sample_n=10000):
        """
        Computes the counterfactual of Xn under the intervention delta.

        Inputs:     Xn: torch.Tensor (N, D) factual
                    sensitive: None or list of int, indices of the sensitive features
                    sample_n: int, number of samples to use for the estimation of the sensitive levels
        Outputs:
                    twins: torch.Tensor (N, D), counterfactual
        """

        # generate interventions values for the sensitive features
        sensitive = self.sensitive if sensitive is None else sensitive

        # Find Sensitive levels
        levels = self.sensitive_levels(sensitive, sample_n)

        # Repeat each Xn sample for number of levels
        intervention = levels.repeat(Xn.shape[0], 1)

        # Create a list to store the repeated tensors
        repeated_tensors = []

        # Repeat the tensor row-wise
        for i in range(Xn.shape[0]):
            repeated_row = Xn[i, :].repeat(levels.shape[0], 1)
            repeated_tensors.append(repeated_row)

        # Concatenate the repeated tensors along the row dimension
        Xn = torch.cat(repeated_tensors, dim=0)

        # create boolean list that is true for sensitive features
        act_index = [i for i in range(Xn.shape[1])]
        soft_label = [not i in sensitive for i in range(Xn.shape[1])]

        # compute counterfactuals
        # Repeat delta for each sensitive levels
        twins_set = self.counterfactual(Xn, intervention, act_index, soft_label)
        return twins_set

    def CAP(self, Xn, sensitive, radius, CAP_n=10000):
        """
        Computes the counterfactual of Xn under the intervention delta.

        Inputs:     Xn: torch.Tensor (N, D) factual
                    sensitive: None or list of int, indices of the sensitive features.
                    delta: torch.Tensor (N, D), intervention values
                    sample_n: int, number of samples to use for the estimation of the sensitive levels
        Outputs:
                    twins: torch.Tensor (N, D), counterfactual
        """

        # generate interventions values for the sensitive features
        sensitive = self.sensitive if sensitive is None else sensitive

        # Distinct values of Xn
        Xn = torch.unique(Xn, dim=0)

        # compute counterfactuals
        twins_set = self.twins(Xn, sensitive)

        # compute semi-latent variables
        q = self.Xn2Q(twins_set)

        # Generate samples from the semi-latent space ball
        slb = self.semi_latent_ball(q, CAP_n, radius, sensitive)

        # Map samples from the semi-latent space to the factual space
        CAP = self.Q2X(slb, sensitive)

        return CAP

    def semi_latent_ball(self, q, n, radius, sensitive=None, Q_metric=l_p):
        """
        Compute samples from semi latent space that are within a ball of radius around q.
        Inputs:     q: torch.Tensor (N, D), point in the semi-latent space
                    n: int, number of samples to generate
                    delta: float, radius of the ball
                    sensitive: None or list of int, indices of the sensitive features
        outputs:    samples: List(torch.Tensor (N, D)), samples from the semi-latent space
        """
        sensitive = self.sensitive if sensitive is None else sensitive
        # complement index of sensitive features
        non_sensitive = list(set(range(q.shape[1])) - set(sensitive))
        d = len(non_sensitive)
        n = int(n)

        center_point = torch.zeros(d)
        ubs = torch.from_numpy(closest_points(center_point, radius, Q_metric, n)).float()
        # ubs = torch.from_numpy(sample_ball(n, d, radius)).float()

        samples = []
        for i in range(q.shape[0]):
            # replicate q[i] n times
            lb = q[i].repeat(n, 1)
            # add unit ball samples to local ball
            lb[:, non_sensitive] += ubs
            samples.append(lb)

        return torch.cat(samples, dim=0)

    def metric(self, x_1, x_2, Q_metric, sensitive=None):
        """
        Compute metric between two samples in the feature space.
        Inputs:ا
            x_1: torch.Tensor (N, D), factual sample 1
            x_2: torch.Tensor (N, D), factual sample 2
            Q_metric: torch.Tensor (N, D), metric in the semi-latent space
            sensitive: None or list of int, indices of the sensitive features
        outputs:
            metric: torch.Tensor (N, ), metric in the feature space
        """
        q_1 = self.X2Q(x_1, sensitive)
        q_2 = self.X2Q(x_2, sensitive)
        return Q_metric(q_1, q_2)

    def positive_pairs(self, X, margin, Q_metric=l_p, sensitive=None):
        sensitive = self.sensitive if sensitive is None else sensitive
        non_sensitive = list(set(range(X.shape[1])) - set(sensitive))
        d = len(non_sensitive)
        num_points = X.shape[0]
        levels = self.sensitive_levels(sensitive)
        random_levels = random_sample(levels, num_points, replace=True)
        center_point = torch.zeros(d)
        close_points, distances = closest_points(center_point, margin, Q_metric, num_points)

        Q = self.X2Q(X, sensitive)
        Q[:, sensitive] = random_levels[:, sensitive]
        Q[:, non_sensitive] += close_points


        labels = torch.ones(Q.shape[0])

        return self.Q2X(Q, sensitive), distances, labels

    def negative_pairs(self, X, margin, Q_metric=l_p, sensitive=None):
        sensitive = self.sensitive if sensitive is None else sensitive
        non_sensitive = list(set(range(X.shape[1])) - set(sensitive))
        d = len(non_sensitive)
        num_points = X.shape[0]

        levels = self.sensitive_levels(sensitive)
        random_levels = random_sample(levels, num_points, replace=True)

        center_point = torch.zeros(d)
        close_points, distances = farthest_points(center_point, margin, Q_metric, num_points)

        Q = self.X2Q(X, sensitive)
        Q[:, sensitive] = random_levels[:, sensitive]
        Q[:, non_sensitive] += close_points

        labels = torch.zeros(Q.shape[0])

        return self.Q2X(Q, sensitive), distances, labels

    def generate_contrastive(self, num_points, margin=1.0, Q_metric=l_p, sensitive=None):
        sensitive = self.sensitive if sensitive is None else sensitive
        n = int(num_points/2)

        X_1, _ = self.generate(n)
        X_1 = self.Xn2X(torch.from_numpy(X_1))
        X_2, _ = self.generate(n)
        X_2 = self.Xn2X(torch.from_numpy(X_2))
        X_pos, d_pos, l_pos = self.positive_pairs(X_1, margin, Q_metric, sensitive)
        X_neg, d_neg, l_neg = self.negative_pairs(X_2, margin, Q_metric, sensitive)

        X1 = torch.cat((X_1, X_2), 0)
        X2 = torch.cat((X_pos, X_neg), 0)
        labels = torch.cat((l_pos, l_neg), 0)
        distances = torch.cat((d_pos, d_neg), 0)

        return {"X1": X1, "X2": X2, "labels": labels, "distances": distances}

    def generate_triplet(self, num_points, margin=.1, Q_metric=l_p, sensitive=None):
        sensitive = self.sensitive if sensitive is None else sensitive
        X_a, _ = self.generate(num_points)
        X_a = self.Xn2X(torch.from_numpy(X_a))
        X_pos, d_pos, _ = self.positive_pairs(X_a, margin, Q_metric, sensitive)
        X_neg, d_neg, _ = self.negative_pairs(X_a, margin, Q_metric, sensitive)
        label = (d_pos + margin < d_neg).to(torch.int)

        return {"X_a": X_a, "X_pos": X_pos, "X_neg": X_neg, "dis_pos": d_pos, "dis_neg": d_neg, "label": label}

    def U2Xn(self, U):
        """
        Mapping from the exogenous variables U to the endogenous X variables, which are standardized

        Inputs:     U: torch.Tensor, shape (N, D)

        Outputs:    Xn: torch.Tensor, shape (N, D), is standardized
        """
        return self.X2Xn(self.U2X(U))

    def Xn2U(self, Xn):
        """
        Mapping from the endogenous variables X (standardized) to the exogenous variables U

        Inputs:     Xn: torch.Tensor, shape (N, D), endogenous variables (features) standardized

        Outputs:    U: torch.Tensor, shape (N, D)
        """
        return self.X2U(self.Xn2X(Xn))

    def Xn2X(self, Xn):
        """
        Transforms the endogenous features to their original form (no longer standardized)

        Inputs:     Xn: torch.Tensor, shape (N, D), features are standardized

        Outputs:    X: torch.Tensor, shape (N, D), features are not standardized
        """
        return Xn * self.std + self.mean

    def X2Xn(self, X):
        """
        Standardizes the endogenous variables X according to self.mean and self.std

        Inputs:     X: torch.Tensor, shape (N, D), features are not standardized

        Outputs:    Xn: torch.Tensor, shape (N, D), features are standardized
        """
        return (X - self.mean) / self.std

    def getActionable(self):
        """ Returns the indices of the actionable features, as a list of ints. """
        return self.actionable

    def get_sensitive(self):
        """ Returns the indices of the sensitive features, as a list of ints. """
        return self.sensitive
