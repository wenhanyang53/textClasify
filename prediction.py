import numpy as np
import copy
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from typing import List
import bisect
import torch
import torch.optim as optim
import logging
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, models, losses, datasets, evaluation
from sentence_transformers.readers import InputExample
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from annoy import AnnoyIndex
import random
from flask import Flask

app = Flask(__name__)

def embedding(dataset, path):
    model = SentenceTransformer(path)
    embeddings = model.encode(dataset)
    print(len(embeddings))
    return embeddings

def annoy_index(embeddings):
    vector_length = embeddings[0].shape[0]
    num_example = len(embeddings)

    tree = AnnoyIndex(vector_length, 'angular')  # Length of item vector that will be indexed
    for i, vector in zip(range(num_example), embeddings):
        tree.add_item(i, vector)

    tree.build(100)

    index = []
    for vector in embeddings:
        index.append(tree.get_nns_by_vector(vector, 100))

    return index

def predict(x_train, y_train):
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(x_train, y_train)

    return knn

def pipeline(dataset, path, num_dataset: int = None, str_to_classify: str = None):
    """
    param dataset:
        dataset with label to train and get predictions
    
    param num_dataset:
        The index number of the dataset that is to be predicted its label
        if None, it will return the accuracy of the model

    param str_to_classify:
        The string that is to be predicted

    """
    if str_to_classify:
        dataset.data.append(str_to_classify)
        embeddings = embedding(dataset.data, path)
    else:
        embeddings = embedding(dataset.data, path)

    index = annoy_index(embeddings)

    if str_to_classify:
        if len(index)-1 in index[-1]:
            index[-1].remove(len(index)-1)
        knn = predict([embeddings[j] for j in index[-1]], [dataset.target[j] for j in index[-1]])
        prediction = knn.predict([embeddings[-1]])
        print(prediction)
        return prediction

    x_train_group = []
    y_train_group = []

    for i in index:
        x_train_group.append([embeddings[j] for j in i])
        y_train_group.append([dataset.target[j] for j in i])

    x_train_group = np.array(x_train_group)
    y_train_group = np.array(y_train_group)

    if type(num_dataset) == int:
        knn = predict(x_train_group[num_dataset], y_train_group[num_dataset])
        prediction = knn.predict([embeddings[num_dataset]])
        print(prediction)
        return prediction
    else:
        y_predict = []
        for i in range(x_train_group.shape[0]):
            knn = predict(x_train_group[i], y_train_group[i])
            y_predict.append(knn.predict([embeddings[i]]))

        num_prediction = 0
        for i in range(x_train_group.shape[0]):
            if y_predict[i] == dataset.target[i]:
                num_prediction += 1

        score = num_prediction / x_train_group.shape[0]
        print(score)
        return score

@app.route('/predict')
def run():
    str_to_classify = 'I want to be elected as a president of France'
    # pipeline(fetch_20newsgroups(subset='train'), "./mymodel")
    # pipeline(fetch_20newsgroups(subset='train'), "distilbert-base-nli-mean-tokens")
    pipeline(dataset=fetch_20newsgroups(subset='train'), path="./mymodel", str_to_classify=str_to_classify)

    return 'OK'

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3000)
