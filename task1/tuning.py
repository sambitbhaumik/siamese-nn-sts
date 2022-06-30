import torch
import optuna
from torch import nn
import logging
from tqdm import tqdm
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from siamese_lstm_attention import SiameseBiLSTMAttention


"""
Script for tuning the model with given hyperparameters. We will use the Optuna hyperparamter optimization framework here.
"""

class Objective(object):
    def __init__(self, sick_data, sick_dataloaders):

        self.sick_data = sick_data
        self.sick_dataloaders = sick_dataloaders

    def __call__(self, trial):

        # initializing certain parameter with a range of values
        output_size = 1
        hidden_size = trial.suggest_int("hidden_size", 32, 128, step=16)
        vocab_size = len(self.sick_data.vocab)
        embedding_size = 300
        embedding_weights = self.sick_data.vocab.vectors
        lstm_layers = trial.suggest_int("lstm_layers", 4, 16, step=4)
        learning_rate = trial.suggest_categorical("learning_rate", [1e-5, 1e-3, 1e-1, 1.0])
        fc_hidden_size = trial.suggest_int("fc_hidden_size",32, 128, step=16)
        max_epochs = 20
        bidirectional = True
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        batch_size = 64

        self_attention_config = {
            "hidden_size": trial.suggest_int("da", 50, 300, step=50),  ## refers to variable 'da' in the ICLR paper
            "output_size": trial.suggest_int("r", 5, 40, step=5),  ## refers to variable 'r' in the ICLR paper
            "penalty": trial.suggest_float("penalty",0.2, 1, log=False, step=0.2),
        }


        siamese_lstm_attention = SiameseBiLSTMAttention(
            batch_size=batch_size,
            output_size=output_size,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            embedding_size=embedding_size,
            embedding_weights=embedding_weights,
            lstm_layers=lstm_layers,
            self_attention_config=self_attention_config,
            fc_hidden_size=fc_hidden_size,
            device=device,
            bidirectional=bidirectional,
        )

        optimizer = torch.optim.Adam(params=siamese_lstm_attention.parameters())
        
        config_dict={
            "device": device,
            "model_name": "siamese_lstm_attention",
            "self_attention_config": self_attention_config,
        }
        
        model=siamese_lstm_attention
        device = config_dict["device"]
        criterion = nn.MSELoss()

        #global max_accuracy
        max_accuracy = 1e-3
        total_val_acc = 0
        
        # iterating though both train and validation dataloaders here. we aim to maximize the validation accuracy
        for epoch in tqdm(range(max_epochs)):

            # TODO implement
            #logging.info("Epoch {}:".format(epoch))

            predictions = list()
            truths = list()
            total_loss = 0

            flag = 0

            for i, data_loader in enumerate(self.sick_dataloaders['train']):
                model.zero_grad()
                sent1_batch, sent2_batch, sent1_len, sent2_len, targets, sent_A, sent_B = data_loader[0:7]
                pred, A_1, A_2 = model(sent1_batch.to(device), sent2_batch.to(device), sent1_len, sent2_len)

                pred = torch.squeeze(pred)
                                    
                A1_penalty = attention_penalty_loss(A_1, config_dict['self_attention_config']['penalty'], device)
                
                A2_penalty = attention_penalty_loss(A_2, config_dict['self_attention_config']['penalty'], device)
            
                pred_loss = criterion(pred.to(device), Variable(targets.float()).to(device))
                loss =  A1_penalty + A2_penalty + pred_loss
                loss.backward()
                optimizer.step()

            ## compute model metrics on dev set
            val_acc = evaluate_dev_set(
                model, self.sick_data, criterion, self.sick_dataloaders, config_dict, device
            )
            if val_acc > max_accuracy:
                max_accuracy = val_acc
            
            ## Pruning unnecessary trials
            trial.report(val_acc, epoch)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return max_accuracy

def tune_model(sick_data, sick_dataloaders):
    # creating optimzation study environment
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(), 
                                                    pruner=optuna.pruners.MedianPruner())
    study.optimize(Objective(sick_data, sick_dataloaders), n_trials=5)

    # returning best accuracy and dictionary of chosen parameters
    return study
    #return study.best_trial.value, study.best_params

def evaluate_dev_set(model, data, criterion, data_loader, config_dict, device):
    """
    Evaluates the model performance on dev data
    """
    #logging.info("Evaluating accuracy on dev set")

    # TODO implement
    predictions = list()
    truths = list()

    for i, data_loader in enumerate(data_loader['validation']):
        sent1_batch, sent2_batch, sent1_len, sent2_len, targets = data_loader[0:5]

        pred, A_1, A_2 = model(sent1_batch.to(device), sent2_batch.to(device), sent1_len, sent2_len)
        pred = torch.squeeze(pred)
        
        predictions += list(pred.detach().cpu().numpy())
        truths += list(targets.numpy())

    # TODO: computing accuracy using sklearn's function
    acc, p_value = pearsonr(truths, predictions)
    return acc

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