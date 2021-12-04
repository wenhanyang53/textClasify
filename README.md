## textClasify
A classifier trained on 20 Newsgroups and fine-tune sBert. You can train it with any text.

## Embedding

The embedding model is trained with distilbert-base-nli-mean-tokens model and TripletLoss loss function. The TripletLoss loss function is customised based on this dataset. 

## Index

Using one of the Approximate Nearest Neighbor algorithm called Annoy to index all the vectors

## Classifier

Using KNN as barebone of the classifier and classify no matter a writing input or an input from the original dataset.

## REST API

An API via '/predict' is built with Flask. Run the python program and request the path.

## Docker

This program can be wrapped in a docker container. Just simply run the docker-compose file.
