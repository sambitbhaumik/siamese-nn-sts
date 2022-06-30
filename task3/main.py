import os
import numpy as np
import pandas as pd
import logging
from preprocess import Preprocess
import torch
import numpy as np
import sts_data
from sts_data import STSData
from importlib import reload
from transformers import BertModel
from train import train_model
from evaluation import evaluate_test_set


def main():
    stopwords_path="stopwords-en.txt"
    dataset_name = "sick"
    batch_size = 64

    dataset_name = "sick"

    sbert_data = STSData(
        dataset_name=dataset_name,
    )

    batch_size = 64
    sbert_dataloaders = sbert_data.get_data_loader(batch_size=batch_size)

    model = BertModel.from_pretrained('bert-base-uncased')

    # setting Adam optimizer with learning rate as specified by SBERT paper
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    max_epochs = 1

    train_acc, train_loss = train_model(model, optimizer, sbert_dataloaders['train'], max_epochs)

    model_path = './sbert_fine_tuned'

    if not os.path.exists(model_path):
        os.mkdir(model_path)

    model.save_pretrained(model_path)

    test_model = BertModel.from_pretrained(model_path)
    evaluate_test_set(test_model,sbert_dataloaders['test'])

if __name__ == "__main__":
    main()
