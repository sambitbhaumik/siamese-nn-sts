import re
import numpy as np
import pandas as pd
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

    def perform_preprocessing(self, df):
        ## TODO normalize text to lower case
        df = df.applymap(lambda s: s.lower() if type(s) == str else s)
        ## TODO remove punctuations
        df = df.applymap(lambda s: re.sub(r'[^\w\s]', '', s) if type(s) == str else s)
        ## TODO remove stopwords
        df = df.applymap(lambda text: " ".join([word for word in str(text).split() if word not in self.stopwords]) if type(text) == str else text)
        ## TODO add any other preprocessing method (if necessary)

        ## removing empty string rows
        ## Reference: https://stackoverflow.com/questions/29314033/drop-rows-containing-empty-cells-from-a-pandas-dataframe
        df = df[df['sentence_A'].astype(bool)]
        df = df[df['sentence_B'].astype(bool)]

        df['relatedness_score'] = pd.to_numeric(df['relatedness_score'])

        #Reference for round up normalization:
        #https://stackoverflow.com/questions/43675014/panda-python-dividing-a-column-by-100-then-rounding-by-2-dp

        df['relatedness_score'] = df['relatedness_score'].div(5).round(2)

        return df
