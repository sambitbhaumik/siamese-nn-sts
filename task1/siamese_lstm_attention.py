import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import similarity_score


"""
Wrapper class using Pytorch nn.Module to create the architecture for our model
Architecture is based on the paper: 
A STRUCTURED SELF-ATTENTIVE SENTENCE EMBEDDING
https://arxiv.org/pdf/1703.03130.pdf
"""


class SiameseBiLSTMAttention(nn.Module):
    def __init__(
        self,
        batch_size,
        output_size,
        hidden_size,
        vocab_size,
        embedding_size,
        embedding_weights,
        lstm_layers,
        device,
        bidirectional,
        self_attention_config,
        fc_hidden_size,
    ):
        super(SiameseBiLSTMAttention, self).__init__()
        """
        Initializes model layers and loads pre-trained embeddings from task 1
        """
        ## model hyper parameters
        self.batch_size = batch_size
        self.output_size = output_size
        self.lstm_hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.lstm_layers = lstm_layers
        self.device = device
        self.bidirectional = bidirectional
        self.fc_hidden_size = fc_hidden_size
        self.lstm_directions = (
            2 if self.bidirectional else 1
        )  ## decide directions based on input flag
        
        pass
        ## model layers
        # TODO initialize the look-up table.
        self.embedding_layer = nn.Embedding(vocab_size, embedding_size)

        # TODO assign the look-up table to the pre-trained fasttext word embeddings.
        self.embedding_layer.weight.data.copy_ = nn.Parameter(embedding_weights.to(self.device), requires_grad=False)

        ## TODO initialize lstm layer
        self.BiLSTM = nn.LSTM(self.embedding_size, self.lstm_hidden_size, num_layers=self.lstm_layers, 
                                bidirectional=self.bidirectional)
        
        ## TODO initialize self attention layers
        self.Attention1 = None
        self.Attention2 = None
        
        ## incase we are using bi-directional lstm we'd have to take care of bi-directional outputs in
        ## subsequent layers
        self.Attention1 = SelfAttention(self.lstm_hidden_size * self.lstm_directions,self_attention_config['hidden_size'],
											self_attention_config['output_size'])
        self.Attention2 = SelfAttention(self.lstm_hidden_size * self.lstm_directions,self_attention_config['hidden_size'],
											self_attention_config['output_size'])

        self.FC1 = nn.Linear(self.lstm_directions * self.lstm_hidden_size * self_attention_config['output_size'], self.fc_hidden_size)
        self.FC2 = nn.Linear(self.lstm_directions * self.lstm_hidden_size * self_attention_config['output_size'], self.fc_hidden_size)

    def init_hidden(self, batch_size):
        """
        Initializes hidden and context weight matrix before each
                forward pass through LSTM
        """
        
        # TODO implement
        if self.bidirectional:
            layers = self.lstm_layers *2
        else:
            layers = self.lstm_layers
        
        hidden = Variable(torch.zeros(layers, batch_size, self.lstm_hidden_size)).to(self.device)
        cell = Variable(torch.zeros(layers, batch_size, self.lstm_hidden_size)).to(self.device)

        # initializing parameters of LSTM as mentioned in the paper
        # we use xavier initialization here
        nn.init.xavier_normal_(hidden)
        nn.init.xavier_normal_(cell)
        return (hidden, cell)
        
    def forward_once(self, batch, lengths):
        """
        Performs the forward pass for each batch
        """

        ## batch shape: (batch_size, seq_len)
        ## embeddings shape: ( batch_size, seq_len, embedding_size)

        # TODO implement

        # fetching word embeddings
        embeddings = self.embedding_layer(batch)
        
        # packing each batch embedding before sending to the LSTM
        packed_embeddings = pack_padded_sequence(embeddings, lengths, batch_first = True, enforce_sorted=False)
        output, (lstm_h, lstm_cell) = self.BiLSTM(packed_embeddings, self.hidden)

        # unpacking the embeddings and adjusting dimension
        pad_output, seq_lens = pad_packed_sequence(output, batch_first = True)
        
        return pad_output

    def forward(self, sent1_batch, sent2_batch, sent1_lengths, sent2_lengths):
        """
        Performs the forward pass for each batch
        """
        ## TODO init context and hidden weights for lstm cell
        self.hidden = self.init_hidden(sent1_batch.size(0))

        # TODO implement forward pass on both sentences. calculate similarity using similarity_score()
        output1 = self.forward_once(sent1_batch, sent1_lengths)

        # each pass for each sentence contains one cycle of self-attention followed by a fully-connected layer 
        # the fully connected layer condenses the sentence embeddings

        # we also construct embedding matrix (M) = A * H, from the self attention paper
        annotation_matrix1 = self.Attention1(output1)
        embedding_matrix1 = torch.bmm(annotation_matrix1, output1)
        output2 = self.forward_once(sent2_batch, sent1_lengths)
        annotation_matrix2 = self.Attention2(output2)
        embedding_matrix2 = torch.bmm(annotation_matrix2, output2)
        s1 = self.FC1(embedding_matrix1.view(embedding_matrix1.size(0), -1))
        s2 = self.FC2(embedding_matrix2.view(embedding_matrix2.size(0), -1))

        # fetching similarity of sentence embeddings
        similarity = similarity_score(s1, s2)

        return similarity, annotation_matrix1, annotation_matrix2

class SelfAttention(nn.Module):
    """
    Implementation of the attention block
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(SelfAttention, self).__init__()
        # TODO implement

        # constructing linear layers with weights analogous to Ws1 and Ws2
        self.layer1 = nn.Linear(input_size, hidden_size, bias=False)
        self.layer2 = nn.Linear(hidden_size, output_size, bias=False)

    ## the forward function would receive lstm's all hidden states as input
    def forward(self, attention_input):
        # TODO implement

        # implementing the attention mechanism
        output = self.layer1(attention_input)
        output = torch.tanh(output)
        output = self.layer2(output)
        output = F.softmax(output.transpose(1,2), dim=2)

        return output