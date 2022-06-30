import torch
from scipy import stats
import logging
from torch import nn
from tqdm import tqdm
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

#logging.basicConfig(level=logging.INFO)

def evaluate_test_set(model, data_loader):
    """
    Evaluates the model performance on dev data
    """
    criterion = nn.MSELoss()
    predictions = list()
    truths = list()
    total_loss = 0

    for batch in tqdm(data_loader):
        inputs_ids_a = batch['sentence_A_input_ids']#.to(device)
        inputs_ids_b = batch['sentence_B_input_ids']#.to(device)
        attention_a = batch['sentence_A_attention_mask']#.to(device)
        attention_b = batch['sentence_B_attention_mask']#.to(device)
        label = batch['relatedness_score']#.to(device)
        u = model(
            inputs_ids_a, attention_mask=attention_a
        )[0]
        v = model(
            inputs_ids_b, attention_mask=attention_b
        )[0]

        u = mean_pool(u, attention_a)
        v = mean_pool(v, attention_b)
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        pred = cos(u,v)
        predictions += list(pred.detach().cpu().numpy())
        truths += list(label.detach().cpu().numpy())

        loss = criterion(pred, label)

        total_loss += loss.item()

    # We use pearson coefficient instead of spearman, to compare performance with our previous models
    test_acc = pearsonr(predictions, truths)

    print("Accuracy: {} ".format(test_acc[0]))

def mean_pool(embedding, mask):
    # before applying pooling we must make sure our mask spans the embeddings
    in_mask = mask.unsqueeze(-1).expand(embedding.size()).float()

    # perform mean-pooling but exclude padding tokens <PAD> (specified by in_mask)
    pooled_output = torch.sum(embedding * in_mask, 1) / torch.clamp(in_mask.sum(1), min=1e-9)
    return pooled_output
