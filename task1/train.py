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
Script for training the neural network and saving the better models 
while monitoring a metric like accuracy etc
"""


def train_model(model, optimizer, dataloader, data, max_epochs, config_dict):
    device = config_dict["device"]
    criterion = nn.MSELoss()
    max_accuracy = 2e-1

    total_val_acc = 0
    val_len = 1

    logging.info("Starting training...")

    for epoch in tqdm(range(max_epochs)):

        # TODO implement
        logging.info("Epoch {}:".format(epoch))

        predictions = list()
        truths = list()
        total_loss = 0

        # iterating through dataloaders
        for i, data_loader in enumerate(dataloader['train']):
            
            model.zero_grad()

            # fetching tensors from batch
            sent1_batch, sent2_batch, sent1_len, sent2_len, targets, sent_A, sent_B = data_loader[0:7]
            
            # performing forward pass and fetching predictions
            pred, A_1, A_2 = model(sent1_batch.to(device), sent2_batch.to(device), sent1_len, sent2_len)
            pred = torch.squeeze(pred)

            # calculating attention penalty for each annotation matrix
            A1_penalty = attention_penalty_loss(A_1, config_dict['self_attention_config']['penalty'], device)
            A2_penalty = attention_penalty_loss(A_2, config_dict['self_attention_config']['penalty'], device)
          
            # calculating MSE loss
            pred_loss = criterion(pred.to(device), Variable(targets.float()).to(device))

            # storing MSE loss and attention penalties as cumulative loss for backpropagation
            loss =  A1_penalty + A2_penalty + pred_loss
            loss.backward()
            optimizer.step()
                
            # storing predictions and truths for later evaluation
            predictions += list(pred.detach().cpu().numpy())
            truths += list(targets.numpy())
            total_loss += pred_loss
            
        # TODO: computing accuracy using sklearn's function

        # Using Pearson Coefficient as evaluation metric and storing result from training process
        acc, p_value = pearsonr(truths, predictions)
        logging.info("Accuracy: {} Training Loss: {}".format(acc, torch.mean(total_loss.float())))

        ## compute model metrics on dev set
        val_acc, val_loss = evaluate_dev_set(
            model, data, criterion, dataloader, config_dict, device
        )

        # storing cumulative validation accuracy
        total_val_acc += val_acc
        val_len += 1

        if val_acc > max_accuracy:
            max_accuracy = val_acc
            logging.info(
                "new model saved"
            )  ## save the model if it is better than the prior best
            torch.save(model.state_dict(), "{}.pth".format(config_dict["model_name"]))
        
        logging.info(
            "Train loss: {} - acc: {} -- Validation loss: {} - acc: {}".format(
                torch.mean(total_loss.data.float()), acc, val_loss, val_acc
            )
        )

    logging.info("Training complete")
    return (total_val_acc/val_len)


def evaluate_dev_set(model, data, criterion, data_loader, config_dict, device):
    """
    Evaluates the model performance on dev data
    """
    logging.info("Evaluating accuracy on dev set")

    # TODO implement
    predictions = list()
    truths = list()
    total_loss = 0

    for i, data_loader in enumerate(data_loader['validation']):
        sent1_batch, sent2_batch, sent1_len, sent2_len, targets = data_loader[0:5]

        pred, A_1, A_2 = model(sent1_batch.to(device), sent2_batch.to(device), sent1_len, sent2_len)
        pred = torch.squeeze(pred)
        
        loss = criterion(pred.to(device), Variable(targets.float()).to(device))
        
        predictions += list(pred.detach().cpu().numpy())
        truths += list(targets.numpy())
        total_loss += loss

    # TODO: computing accuracy using sklearn's function
    # Using Pearson Coefficient as evaluation metric

    acc, p_value = pearsonr(truths, predictions)
    return acc, torch.mean(total_loss.float())

def attention_penalty_loss(annotation_weight_matrix, penalty_coef, device):
    """
    This function computes the loss from annotation/attention matrix
    to reduce redundancy in annotation matrix and for attention
    to focus on different parts of the sequence corresponding to the
    penalty term 'P' in the ICLR paper
    ----------------------------------
    'annotation_weight_matrix' refers to matrix 'A' in the ICLR paper
    annotation_weight_matrix shape: (batch_size, attention_out, seq_len)
    """
    batch_size, attention_out_size = annotation_weight_matrix.size(0), annotation_weight_matrix.size(1)
    
    # TODO implement

    # implementation of attention penalty term (with batch normalization)
    A_transpose = annotation_weight_matrix.transpose(1,2)
    annotation = torch.matmul(annotation_weight_matrix, A_transpose)

    identity = Variable(torch.eye(annotation_weight_matrix.size(1)).unsqueeze(0).expand(batch_size, attention_out_size, attention_out_size)).to(device)
    annotation_mul_difference = annotation - identity

    penalization_term = frobenius_norm(annotation_mul_difference)
    new_loss = torch.as_tensor((penalty_coef * penalization_term)/batch_size)
    
    return new_loss

def frobenius_norm(annotation_mul_difference):
    """
    Computes the frobenius norm of the annotation_mul_difference input as matrix
    """
    # TODO implement
    a_i = torch.sum(annotation_mul_difference**2, 1)
    a_j = torch.sum(a_i, 1)

    norm = torch.sum(a_j ** 0.5)

    return norm