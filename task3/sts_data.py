import pandas as pd
from preprocess import Preprocess
import logging
import torch
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

logging.basicConfig(level=logging.INFO)

"""
For loading STS data loading and preprocessing
"""


class STSData:
    def __init__(
        self,
        dataset_name,
        stopwords_path="stopwords-en.txt",
        max_sequence_len=20,
    ):
        """
        Loads data into memory
        """
        self.max_sequence_len = max_sequence_len
        self.dataset_name = dataset_name
        ## load data file into memory
        self.load_data(dataset_name, stopwords_path)

    def load_data(self, dataset_name, stpwds_file_path):
        """
        Reads data set file from disk to memory using hugginface dataset downloader
        """
        logging.info("loading and preprocessing data...")

        ## TODO load datasets

        #Loading datasets by their respective splits using hugginface datasets

        train_data_set = load_dataset(dataset_name, split="train")
        val_data_set = load_dataset(dataset_name, split="validation")
        test_data_set = load_dataset(dataset_name, split="test")

        columns = ["sentence_A","sentence_B","relatedness_score"]
        # converting to pandas dataframe to ease preprocessing

        train_df = pd.DataFrame(train_data_set)[columns]
        val_df = pd.DataFrame(val_data_set)[columns]
        test_df = pd.DataFrame(test_data_set)[columns]

        # dataframe preprocessing
        pre = Preprocess(stpwds_file_path)
        self.train_df = pre.perform_preprocessing(train_df)
        self.val_df = pre.perform_preprocessing(val_df)
        self.test_df = pre.perform_preprocessing(test_df)

        # reconverting to hugginface dataset from pandas dataframe
        self.train_df = Dataset.from_pandas(self.train_df)
        self.train_df = self.train_df.remove_columns(['__index_level_0__'])
        self.test_df = Dataset.from_pandas(self.test_df)
        self.test_df = self.test_df.remove_columns(['__index_level_0__'])
        self.val_df = Dataset.from_pandas(self.val_df)
        self.val_df = self.val_df.remove_columns(['__index_level_0__'])

        logging.info("reading and preprocessing data completed...")

    def get_data_loader(self, batch_size=8):
        # TODO implement

        ## Here we create train and test dataloaders
        df_dict = {'train': self.train_df, 'test': self.test_df}

        for key, df in df_dict.items():

            # We perform the following here:
            # 1. Iterate through dataset columns, tokenize and numericalize both the sentences using the BertTokenizer
            # 2. Rename new tokenizer items to adhere to both sentences (helps us in data loading)
            all_cols = ['relatedness_score']
            dataset = df
            for sentence in ['sentence_A', 'sentence_B']:
                dataset = dataset.map(
                    lambda x: tokenizer(
                        x[sentence], max_length=self.max_sequence_len, padding='max_length',
                        truncation=True
                    ), batched=True
                )

                # Reference for new items: https://huggingface.co/docs/transformers/v4.17.0/en/preprocessing#tokenize
                for new_col in ['input_ids', 'attention_mask']:
                    dataset = dataset.rename_column(
                        new_col, sentence+'_'+new_col
                    )
                    all_cols.append(sentence+'_'+new_col)

            # set_format is used to cast dataset columns to PyTorch tensors with provided column names
            # https://huggingface.co/docs/datasets/master/en/package_reference/main_classes#datasets.Dataset.set_format

            dataset.set_format(type='torch', columns=all_cols)

            # saving respective Dataloaders

            if(key == 'train'):
                train_loader = torch.utils.data.DataLoader(
                    dataset, batch_size=batch_size, shuffle=True
                )

            else:
                test_loader = torch.utils.data.DataLoader(
                    dataset, batch_size=batch_size, shuffle=True
                )

        data_loader_dict = {'train': train_loader, 'test': test_loader}

        logging.info("creating dataloaders completed...")

        return data_loader_dict
