import torch
from scipy import stats
import logging
from torch import nn
from tqdm import tqdm
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import mean_squared_error

logging.basicConfig(level=logging.INFO)

def evaluate_test_set(model, data_loader, config_dict):
    """
    Evaluates the model performance on dev data
    """
    logging.info("Evaluating accuracy on test set")

    # TODO implement
    predictions = list()
    truths = list()
    total_loss = 0
    device = config_dict["device"]
    criterion = nn.MSELoss()

    for i, data_loader in enumerate(data_loader['test']):
        sent1_batch, sent2_batch, sent1_len, sent2_len, targets = data_loader[0:5]

        pred, A_1, A_2 = model(sent1_batch.to(device), sent2_batch.to(device), sent1_len, sent2_len)
        pred = torch.squeeze(pred)

        loss = criterion(pred.to(device), targets.float().to(device))

        predictions += list(pred.detach().cpu().numpy())
        truths += list(targets.numpy())
        total_loss += loss
    
    #predictions = torch.tensor(predictions) 
    #truths = torch.tensor(truths)

    #correct_pred = torch.abs(predictions-truths) > 0.05
    #truth_base = torch.tensor([True]*correct_pred.size(0))   
    # TODO: computing accuracy using sklearn's function
    acc, p_value = stats.pearsonr(truths, predictions)
    
    print("Accuracy: {} Test Loss: {}".format(acc, torch.mean(total_loss.float())))