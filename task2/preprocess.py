import re
import numpy as np
"""
Performs basic text cleansing on the unstructured field
"""


class Preprocess:
    def __init__(self, stpwds_file_path):
        """
        Initializes regex patterns and loads stopwords
        """
        # TODO implement
        with open(stpwds_file_path, 'r', encoding="utf8") as file:
            self.stopwords = [line.rstrip() for line in file]

    def perform_preprocessing(self, df, columns_mapping):
        ## TODO normalize text to lower case
        df = df.applymap(lambda s: s.lower() if type(s) == str else s)
        ## TODO remove punctuations
        df = df.applymap(lambda s: re.sub(r'[^\w\s]', '', s) if type(s) == str else s)
        ## TODO remove stopwords
        df = df.applymap(lambda text: " ".join([word for word in str(text).split() if word not in self.stopwords]) if type(text) == str else text)
        ## TODO add any other preprocessing method (if necessary)
        
        ## removing empty string rows
        df = df[df['sentence_A'].astype(bool)]
        df = df[df['sentence_B'].astype(bool)]

        return df
