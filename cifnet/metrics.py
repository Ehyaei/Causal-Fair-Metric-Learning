""" Metrics for evaluating the performance of the model."""
import torch
from sklearn.metrics import matthews_corrcoef


def RMSE(x, y):
    return torch.sqrt(torch.mean((x - y) ** 2))


def mpe(x, y):
    return torch.mean(torch.abs((x - y) / y))

def Accuracy(x, y):
    return torch.mean((x == y).float())


def MCC(x, y):
    return matthews_corrcoef(x, y)


def performance_metrics(model, test_dict, data_type):
    """
    Compute the performance metrics of the model on the test set.
    :param model:  model to evaluate
    :param test_dict:  test dictionary
    :param data_type: type of data (contrastive or triplet)
    :return: accuracy, rmse, r2, mcc
    """
    model.eval()
    with torch.no_grad():
        if data_type == 'contrastive':
            label_pred, distance_pred = model.predict(test_dict)
            label_true = test_dict['labels']
            distance_true = test_dict['distances']
        elif data_type == 'triplet':
            distance_positive, distance_negative, label_pred = model.predict(test_dict)
            label_true = test_dict['label']
            distance_pred = torch.cat((distance_positive, distance_negative), 0)
            distance_true = torch.cat((test_dict['d_1'], test_dict['d_2']), 0)
        else:
            raise AssertionError

        acc = Accuracy(label_pred, label_true)
        mcc = MCC(label_pred, label_true)
        rmse = RMSE(distance_pred, distance_true)
        ars = mpe(distance_pred, distance_true)

    return acc, rmse, ars, mcc
