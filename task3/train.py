import torch
from torch import nn
import logging
from tqdm import tqdm
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

logging.basicConfig(level=logging.INFO)

"""
Script for constructing the SBERT model by constructing siamese BERT networks.
Fine-tuning the pre-trained model and saving the better models
while monitoring a metric like accuracy etc
"""

def train_model(model, optimizer, loader, max_epochs):
    criterion = nn.MSELoss()
    predictions = list()
    truths = list()
    total_loss = 0

    for epoch in range(max_epochs):
        model.train()
        for batch in tqdm(loader):
            # preventing gradient accumulation for BERT
            optimizer.zero_grad()


            inputs_ids_a = batch['sentence_A_input_ids']#.to(device)
            inputs_ids_b = batch['sentence_B_input_ids']#.to(device)
            attention_a = batch['sentence_A_attention_mask']#.to(device)
            attention_b = batch['sentence_B_attention_mask']#.to(device)
            label = batch['relatedness_score']#.to(device)

            # extract words embeddings from BERT
            # here we share the BERT model for both the sentences, thereby creating
            # a siamese network for SBERT construction

            # Here we pass the attention_mask from the tokenizer which notes the
            # position of the <pad> tokens during tokenization

            # extracting sentence_A embeddings
            u = model(inputs_ids_a, attention_mask=attention_a)[0]
            # extracting sentence_B embeddings
            v = model(inputs_ids_b, attention_mask=attention_b)[0]

            # Pool the word embeddings across their token vectors
            u = mean_pool(u, attention_a)
            v = mean_pool(v, attention_b)

            # We use CosineSimilarity loss to form the SBERT Similarity layer
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)

            pred = cos(u,v)
            predictions += list(pred.detach().cpu().numpy())
            truths += list(label.detach().cpu().numpy())

            # MSE Loss
            loss = criterion(pred, label)

            total_loss += loss.item()

            loss.backward()
            optimizer.step()

    # We use pearson coefficient instead of the SBERT paper's use of spearman, to compare performance with our previous models
    acc = pearsonr(predictions, truths)
    logging.info("Accuracy: {}".format(acc))

    return acc[0], total_loss

def mean_pool(embedding, mask):
    # before applying pooling we must make sure our mask spans the embeddings
    in_mask = mask.unsqueeze(-1).expand(embedding.size()).float()

    # perform mean-pooling but exclude padding tokens <PAD> (specified by in_mask)
    pooled_output = torch.sum(embedding * in_mask, 1) / torch.clamp(in_mask.sum(1), min=1e-9)
    return pooled_output
