import pandas as pd
from preprocess import Preprocess
import logging
import torch
from dataset import STSDataset
from datasets import load_dataset
import torchtext
from torch.utils.data import DataLoader
import spacy
import numpy as np
from torchtext.legacy import data
spacy.load('en_core_web_sm')

logging.basicConfig(level=logging.INFO)

"""
For loading STS data loading and preprocessing
"""


class STSData:
    def __init__(
        self,
        dataset_name,
        columns_mapping,
        stopwords_path="stopwords-en.txt",
        model_name="lstm",
        max_sequence_len=20,
        normalization_const=5.0,
        normalize_labels=False,
    ):
        """
        Loads data into memory and create vocabulary from text field.
        """
        self.normalization_const = normalization_const
        self.normalize_labels = normalize_labels
        self.model_name = model_name
        self.max_sequence_len = max_sequence_len
        self.dataset_name = dataset_name
        ## load data file into memory
        self.load_data(dataset_name, columns_mapping, stopwords_path)
        self.columns_mapping = columns_mapping
        ## create vocabulary
        self.create_vocab()

    def load_data(self, dataset_name, columns_mapping, stpwds_file_path):
        """
        Reads data set file from disk to memory using pandas
        """
        logging.info("loading and preprocessing data...")

        ## TODO load datasets
        #Loading

        import pandas as pd
        df = pd.read_csv('SICK\SICK.txt', sep="\t")
        
        train_mask = df['SemEval_set'] == 'TRAIN'
        val_mask = df['SemEval_set'] == 'TRIAL'
        test_mask = df['SemEval_set'] == 'TEST'
        train_data_set = df[train_mask]
        val_data_set = df[val_mask]
        test_data_set = df[test_mask]

        columns = ["sentence_A","sentence_B","relatedness_score"]
        train_df = train_data_set[columns]
        val_df = val_data_set[columns]
        test_df = test_data_set[columns]

        #Preprocessing
        pre = Preprocess(stpwds_file_path)
        self.train_df = pre.perform_preprocessing(train_df, columns_mapping)
        self.val_df = pre.perform_preprocessing(val_df, columns_mapping)
        self.test_df = pre.perform_preprocessing(test_df, columns_mapping)

        logging.info("reading and preprocessing data completed...")

    def create_vocab(self):
        """
        Creates vocabulary over entire text data field.
        """
        logging.info("creating vocabulary...")

        # TODO create vocabulary
        voc_df = self.train_df
        voc_df["sentences"] = voc_df[['sentence_A', 'sentence_B']].apply(lambda x: ' '.join(x), axis=1)
        voc_df = voc_df[["sentences","relatedness_score"]]

        sent = data.Field(sequential=True,
                               tokenize=data.get_tokenizer('spacy',language='en_core_web_sm'),
                               use_vocab=True)
        token = voc_df['sentences'].apply( lambda x: sent.preprocess(x))
        sent.build_vocab(
            token,
            vectors='fasttext.simple.300d'
        )
        self.vocab = sent.vocab

        logging.info("creating vocabulary completed...")

    def min_max_scaling(self, series):
        return (series - series.min()) / (series.max() - series.min())
        #return ((series)/5.0)

    def data2tensors(self, data):
        """
        Converts raw data sequences into vectorized sequences as tensors
        """
        pass
        # TODO implement

    def get_data_loader(self, batch_size=8):
        pass
        # TODO implement

        df_dict = {'train': self.train_df, 'validation': self.val_df, 'test': self.test_df}

        for key, df in df_dict.items():
            df['relatedness_score'] = pd.to_numeric(df['relatedness_score'])
            if(self.normalize_labels == True):
                df['relatedness_score'] = self.min_max_scaling(df['relatedness_score'])

            sent_A_raw, sent_B_raw = df['sentence_A'].values, df['sentence_B'].values

            sent_A = list(map(self.vectorize_sequence, sent_A_raw))
            sent_B = list(map(self.vectorize_sequence, sent_B_raw))

            sent1_length_tensor = torch.LongTensor([len(w) for w in sent_A])
            sent2_length_tensor = torch.LongTensor([len(w) for w in sent_B])

            sent_A = self.pad_sequences(sent_A, self.max_sequence_len)
            sent_B = self.pad_sequences(sent_B, self.max_sequence_len)

            sent1_tensor, sent2_tensor = torch.LongTensor(sent_A), torch.LongTensor(sent_B)

            target = df['relatedness_score'].values
            target_tensor = torch.tensor(target)

            if(key == 'train'):
                train_data = STSDataset(sent1_tensor,sent2_tensor,target_tensor,sent1_length_tensor,sent2_length_tensor,sent_A_raw, sent_B_raw)
            elif(key == 'validation'):
                val_data = STSDataset(sent1_tensor,sent2_tensor,target_tensor,sent1_length_tensor,sent2_length_tensor,sent_A_raw, sent_B_raw)
            else:
                test_data = STSDataset(sent1_tensor,sent2_tensor,target_tensor,sent1_length_tensor,sent2_length_tensor,sent_A_raw, sent_B_raw)

        logging.info("creating STSDataset completed...")
        train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
        val_loader = DataLoader(val_data, shuffle=True, batch_size=batch_size)
        test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

        data_loader_dict = {'train': train_loader, 'validation': val_loader, 'test': test_loader}
        logging.info("creating dataloaders completed...")
        return data_loader_dict

    def sort_batch(self, batch, targets, lengths):
        """
        Sorts the data, lengths and target tensors based on the lengths
        of the sequences from longest to shortest in batch
        """
        sents1_lengths, perm_idx = lengths.sort(0, descending=True)
        sequence_tensor = batch[perm_idx]
        target_tensor = targets[perm_idx]
        return sequence_tensor.transpose(0, 1), target_tensor, sents1_lengths

    def vectorize_sequence(self, sentence):
        """
        Replaces tokens with their indices in vocabulary
        """
        pass
        # TODO implement
        return [self.vocab[w] for w in sentence.split()]

    def pad_sequences(self, vectorized_sents, sents_lengths):
        """
        Pads zeros at the end of each sequence in data tensor till max
        length of sequence in that batch
        """
        pass
        # TODO implement
        return [np.pad(sent, (0, sents_lengths - len(sent))) for sent in vectorized_sents]
