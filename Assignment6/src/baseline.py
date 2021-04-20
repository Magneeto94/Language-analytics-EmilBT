'''
----------------------------IMPORTING PAKEGES:----------------
'''
# system tools
import os
import sys
sys.path.append(os.path.join(".."))

# pandas, numpy, gensim
import pandas as pd
import numpy as np
import gensim.downloader

# import my classifier utility functions - see the Github repo!
import utils.classifier_utils as clf

# Machine learning stuff
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn import metrics

# tools from tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Embedding, 
                                     Flatten, GlobalMaxPool1D, Conv1D)
from tensorflow.keras.optimizers import SGD, Adam #Adam = optimerings algoritme
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.regularizers import L2

# matplotlib
import matplotlib.pyplot as plt


'''
---------------------------------DEFINING PLOTFUNCTION:-------------------
'''

def plot_history(H, epochs):
    """
    Utility function for plotting model history using matplotlib
    
    H: model history 
    epochs: number of epochs for which the model was trained
    """
    plt.style.use("fivethirtyeight")
    plt.figure()
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()
    

def create_embedding_matrix(filepath, word_index, embedding_dim):
    """ 
    A helper function to read in saved GloVe embeddings and create an embedding matrix
    
    filepath: path to GloVe embedding
    word_index: indices from keras Tokenizer
    embedding_dim: dimensions of keras embedding layer
    """
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath) as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word] 
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix



#Deffining main function
def main():
    
    #Deffining filepath
    filepath = '../data/Game_of_Thrones_Script.csv'
    
    #Reading csv file as data frame.
    dataframe = pd.read_csv(filepath)
    
    #Creating np arrays based on the dataframe with the sentence, season and names data.
    sentences = dataframe["Sentence"].values
    seasons = dataframe["Season"].values
    names = dataframe["Name"].values
    
    #Creating test and training data
    X_train, X_test, y_train, y_test = train_test_split(sentences,
                                                        seasons,
                                                        test_size=0.25,
                                                        random_state=42)
    
    
    vectorizer = CountVectorizer()
    
    
    
    # First we do it for our training data...
    X_train_feats = vectorizer.fit_transform(X_train)
    #... then we do it for our test data
    X_test_feats = vectorizer.transform(X_test)
    # We can also create a list of the feature names. 
    feature_names = vectorizer.get_feature_names()
    
    
    #EBT: fitting the model.
    classifier = LogisticRegression(random_state=42).fit(X_train_feats, y_train)
    
    y_pred = classifier.predict(X_test_feats)
    
    
    classifier_metrics = metrics.classification_report(y_test, y_pred)
    print(classifier_metrics)
    
    
    
    clf.plot_cm(y_test, y_pred, normalized=True)
    plt.savefig("../output_baseline/actual_and_predicted.png")
    
    
    
    # Vectorize full dataset
    X_vect = vectorizer.fit_transform(sentences)

    # initialise cross-validation method
    title = "Learning Curves (Logistic Regression)"
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    # run on data
    model = LogisticRegression(random_state=42)
    clf.plot_learning_curve(model, title, X_vect, seasons, cv=cv, n_jobs=4)
    plt.savefig("../output_baseline/three_plots.png")
    
    
if __name__ =='__main__':
    main()