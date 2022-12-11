###### LOADING PACKAGES ######
# system tools
import os

# pytorch
os.system('pip install --upgrade pip datasets torch')
import torch
import torch.nn as nn

# data processing
import pandas as pd
import numpy as np

# huggingface datasets
from datasets import load_dataset

# scikit learn tools
from sklearn.model_selection import train_test_split
from datasets import Dataset
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

# plotting tools
import matplotlib.pyplot as plt


###### DEFINING LOGISTIC REGRESSION MODEL CLASS ######
# Logistic regression model
class LogReg(nn.Module):
    """ class initializing a simple logistic regression classifier
    """
    def __init__(self, n_input_features):
         super().__init__()
         self.linear = torch.nn.Linear(n_input_features, 1)

    def forward(self, x):
        x = self.linear(x)
        y_pred = torch.sigmoid(x)
        return y_pred

# Neural network
class NeuralNetwork(nn.Module):
    def __init__(self, n_input_features):
        super().__init__()
        self.linear1 = nn.Linear(n_input_features, 100)
        self.linear2 = nn.Linear(100, 10)
        self.linear3 = nn.Linear(10, 1)

    def forward(self, x):
        leaky_relu = nn.LeakyReLU(0.2)
        # Linear -> ReLU
        x = self.linear1(x)
        x = leaky_relu(x)
        # Linear -> ReLU
        x = self.linear2(x)
        x = leaky_relu(x)
        # Linear -> Sigmoid
        x = self.linear3(x)
        y_pred = torch.sigmoid(x)
        return y_pred


###### LOADING THE DATA ######
def SplitData(data_path):
    """ A function that loads the "raw" dataframe and splits it into three subsets:
    training, validation, and test data.
    NB: The data is not really raw - the TalkBank data has been preprocessed - see xxx.
    """
    # Import the dataframe
    data = pd.read_csv(data_path)

    # Split dataset into train, test, val (70, 15, 15)
    train, test = train_test_split(data, test_size=0.15)
    train, val = train_test_split(train, test_size=0.15)

    # Turning the split dataframes into dicts
    train = Dataset.from_dict(train)
    val = Dataset.from_dict(val)
    test = Dataset.from_dict(test)
    
    return train, val, test


###### LOADING THE DATA ######
def Vectorize(data_path): # xxx rename to TF_IDF() or BOW() one you have decided whether to use one or the other vectorizer.
    """ This function creates a TF-IDF model of the training, 
    validation, and test sets using sklearn's TfidfVectorizer.
    """

    # Load and split the data
    train, val, test = SplitData(data_path)

    # Define vectorizer
    vectorizer = TfidfVectorizer(ngram_range=(1,2),
                            #lowercase=True, 
                            #stop_words="english", 
                            #max_df=0.9, 
                            #min_df=0.1,
                            max_features=500)
    
    
    # Vectorizing the datasets
    train_vect = vectorizer.fit_transform(train["text"])
    val_vect = vectorizer.transform(val["text"])
    test_vect = vectorizer.transform(test["text"])

    # Turning the vectorized data into tensors
    ## Training data:
    train_vect = torch.tensor(train_vect.toarray(), dtype=torch.float)
    train_label = torch.tensor(list(train["label"]), dtype=torch.float)
    train_label = train_label.view(train_label.shape[0], 1)

    # Validation data:
    val_vect = torch.tensor(val_vect.toarray(), dtype=torch.float)
    val_label = torch.tensor(list(val["label"]), dtype=torch.float)
    val_label = val_label.view(val_label.shape[0], 1)

    # Test data
    test_vect = torch.tensor(test_vect.toarray(), dtype=torch.float)
    test_label = torch.tensor(list(test["label"]), dtype=torch.float)
    test_label = test_label.view(test_label.shape[0], 1)

    return train_vect, train_label, val_vect, val_label, test_vect, test_label



###### SET MODEL PARAMETERS ######
def InitializeModel(train_vect, classifier, learning_rate):

    n_samples, n_features = train_vect.shape

    # initializing the chosen classifier
    if classifier == 'nn':
        model = NeuralNetwork(n_input_features=n_features)
    
    elif classifier == 'logreg':
        model = LogReg(n_input_features=n_features)

    else:
        print('Not valid classifier - please try again')
        exit()

    # define loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(),
                                lr = learning_rate) # 1e-3
    
    return n_features, criterion, optimizer, model


###### TRAINING THE MODEL ######
def Train(data_path, epochs, classifier, learning_rate): # xxx plot = True
    
    print("[INFO:] Training classifier...")

    # Initialize the model
    n_features, criterion, optimizer, model = InitializeModel(train_vect, classifier, learning_rate)

    # for plotting
    train_loss_history = []
    val_loss_history = []

    # loop for epochs
    for epoch in range(epochs):
    
        # forward
        y_hat = model(train_vect)

        # backward
        loss = criterion(y_hat, train_label)
        train_loss_history.append(loss)

        # backpropagation
        loss.backward()
    
        # take step, reset
        optimizer.step()
        optimizer.zero_grad()
    
        # Validation Loop 
        with torch.no_grad(): 
            # set to eval mode
            model.eval() 

            # make predictions
            predicted_outputs = model(val_vect) 

            # metrics
            val_loss = criterion(predicted_outputs, val_label) 

            # append
            val_loss_history.append(val_loss) 

        # some print to see that it is running
        if (epoch + 1) % 100 == 0:
            print(f"epoch: {epoch+1}, loss = {loss.item():.4f}")

    print("[INFO:] Finished traning!")

    # Plot the training and valudation loss curves
    train_loss = [val.item() for val in train_loss_history]
    val_loss = [val.item() for val in val_loss_history]

    fig, ax = plt.subplots()
    ax.plot(train_loss)
    ax.plot(val_loss)
    plt.savefig(os.path.join("/work", "exam", "ASD_classification", "out", "loss_curve_" + classifier + ".png"))

    return model, n_features


###### EVALUATING THE MODEL ######
def Test(n_features, model, test_vect, test_label, classifier):
    """ Function to evaluate model on test set
    Args:
        n_features (int): number of features
    """

    # Write classification report
    predicted = model(test_vect).detach().numpy()
    output = classification_report(test_label, 
                                np.where(predicted > 0.5, 1, 0),
                                target_names = ["TD", "ASD"])

    print(output)

    # Save classification report
    with open(os.path.join("/work", "exam", "ASD_classification", "out", "classification_report_" + classifier + ".txt"), 'w') as f:
        f.write(output)
        f.write(f"Epochs = {epochs}      ")
        f.write(f"Learning rate = {learning_rate}")


###### MAIN ######
if __name__ == "__main__": 

    # Receiving user input
    classifier = input("It is time select your classifier! \
    Type logreg for logistic regression or nn for a simple neural network \
    with one hidden layers containing 30 nodes: ")
    print(f'\n Your chosen classifier {classifier} will be fitted shortly')

    # User defined variables
    data_path = "/work/exam/ASD_classification/data/dataframes/data_eigstig_age3_text_label.csv"
    epochs = 1000
    learning_rate = 1e-4

    # Splitting into separate datasets
    train_vect, train_label, val_vect, val_label, test_vect, test_label = Vectorize(data_path)

    # training the model
    model, n_features = Train(data_path, epochs, classifier, learning_rate)

    Test(n_features, model, test_vect, test_label, classifier)