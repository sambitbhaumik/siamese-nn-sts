# Neural Networks Final Project (NLP)
Neural Networks: Theory and Implementation (WS 2021/22, Universität des Saarlandes)

Project files contain PyTorch implementations for Siamese BiLSTM models for Semantic Text Similarity on the SICK Dataset using FastText embeddings. Also contains Siamese BiLSTM-Transformer Encoder and SBERT fine-tuning implementatons on the STS Data tasks.

## Task 1
Combine architectures from two papers (Mueller & Thyagarajan, 2016) and (Lin et al., 2017) to solve the STS task. This serves as the baseline architecture.

![Untitled%20Diagram.drawio%20%281%29.png](https://raw.githubusercontent.com/shahrukhx01/ocr-test/main/download.png)

## Task 2
Improving on the baseline architecture from task 1 by implementing the Transformer Encoder (Vaswani et al., 2017) from scratch and adding to the architecture
from Task 1.

## Task 3
Miscellaneous task to achieve near SOTA results on STS task. We use a pretrained SBERT model and fine-tune it on the current STS Task.

### Resources

Dataset:
[SICK Dataset](https://huggingface.co/datasets/sick)

SBERT Pre-Trained Model:
[SentenceTransformers](https://www.sbert.net)

Spacy: (https://spacy.io)
