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
from sklearn.feature_extraction.text import CountVectorizer #xxx

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
        self.linear1 = nn.Linear(n_input_features, 32) #xxx
        self.linear2 = nn.Linear(32, 1)
        #self.linear3 = nn.Linear(10, 1)

    def forward(self, x):
        leaky_relu = nn.LeakyReLU(0.1) #xxx Ross had 0.2
        # Linear -> ReLU
        x = self.linear1(x)
        x = leaky_relu(x)
        # Linear -> ReLU
        x = self.linear2(x)
        #x = leaky_relu(x)
        # Linear -> Sigmoid
        #x = self.linear3(x)
        y_pred = torch.sigmoid(x)
        return y_pred

# class NeuralNetwork(nn.Module):
#     def __init__(self, n_input_features):
#         super().__init__()
#         self.linear1 = nn.Linear(n_input_features, 20) #xxx
#         self.linear2 = nn.Linear(20, 20)
#         self.linear3 = nn.Linear(20, 1)

#     def forward(self, x):
#         leaky_relu = nn.LeakyReLU(0.1) #xxx Ross had 0.2
#         # Linear -> ReLU
#         x = self.linear1(x)
#         x = leaky_relu(x)
#         # Linear -> ReLU
#         x = self.linear2(x)
#         x = leaky_relu(x)
#         # Linear -> Sigmoid
#         x = self.linear3(x)
#         y_pred = torch.sigmoid(x)
#         return y_pred


###### LOADING THE DATA ######
def SplitData(data_path):
    """ A function that loads the "raw" dataframe and splits it into three subsets:
    training, validation, and test data.
    NB: The data is not really raw - the TalkBank data has been preprocessed - see xxx.
    """
    # # Import the dataframe
    data = pd.read_csv(data_path)

    # # Split dataset into train, test, val (70, 15, 15)
    train, test = train_test_split(data, test_size=0.15)
    train, val = train_test_split(train, test_size=0.15)

    # Turning the split dataframes into dicts
    train = Dataset.from_dict(train)
    val = Dataset.from_dict(val)
    test = Dataset.from_dict(test)

    # from datasets import load_dataset

    # dataset = load_dataset("rotten_tomatoes")

    # obtain train, val and test splits
    # train = dataset["train"]
    # val = dataset["validation"]
    # test = dataset["test"]
    
    return train, val, test

###### LOADING THE DATA ######
def Vectorize(data_path): # xxx rename to TF_IDF() or BOW() one you have decided whether to use one or the other vectorizer.
    """ This function creates a TF-IDF model of the training, 
    validation, and test sets using sklearn's TfidfVectorizer.
    """

    # Load and split the data
    train, val, test = SplitData(data_path)

    # Define vectorizer
    vectorizer = TfidfVectorizer(ngram_range=(1,1), # xxx default is 1,1 which means that all we want from the vocab is for it to consist of 1 word tokens and 2 word tokens. e.g. good but also not good
    # try to comment out the ngram and just use the default. Experiment with this.
                            lowercase=True, 
                            #stop_words="english", 
                            max_df=0.95, 
                            min_df=0.05,
                            max_features=500) #reduce noise in the data
    
    
    # Vectorizing the datasets
    x_train = vectorizer.fit_transform(train["text"])
    x_val = vectorizer.transform(val["text"])
    x_test = vectorizer.transform(test["text"])

    # Turning the vectorized data into tensors
    ## Training data:
    x_train = torch.tensor(x_train.toarray(), dtype=torch.float)
    y_train = torch.tensor(list(train["label"]), dtype=torch.float)
    y_train = y_train.view(y_train.shape[0], 1)

    # Validation data:
    x_val = torch.tensor(x_val.toarray(), dtype=torch.float)
    y_val = torch.tensor(list(val["label"]), dtype=torch.float)
    y_val = y_val.view(y_val.shape[0], 1)

    # Test data
    x_test = torch.tensor(x_test.toarray(), dtype=torch.float)
    y_test = torch.tensor(list(test["label"]), dtype=torch.float)
    y_test = y_test.view(y_test.shape[0], 1)

    return x_train, y_train, x_val, y_val, x_test, y_test



###### SET MODEL PARAMETERS ######
def InitializeModel(x_train, classifier, learning_rate):

    n_samples, n_features = x_train.shape

    # initializing the chosen classifier
    if classifier == 'nn':
        model = NeuralNetwork(n_input_features=n_features)
    
    elif classifier == 'lr':
        model = LogReg(n_input_features=n_features)

    else:
        print('Not valid classifier - please try again')
        exit()

    # define loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(),
                                lr = learning_rate)
    
    return n_features, criterion, optimizer, model


###### TRAINING THE MODEL ######
def Train(data_path, epochs, classifier, learning_rate): # xxx plot = True
    
    print("[INFO:] Training classifier...")

    # Initialize the model
    n_features, criterion, optimizer, model = InitializeModel(x_train, classifier, learning_rate)

    # for plotting
    train_loss_history = []
    val_loss_history = []

    # loop for epochs
    for epoch in range(epochs):
    
        # forward
        y_hat = model(x_train)

        # backward
        loss = criterion(y_hat, y_train)
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
            y_val_hat = model(x_val) # predicted outputs

            # metrics
            val_loss = criterion(y_val_hat, y_val) 

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
    ax.plot(train_loss, label = 'train')
    ax.set_title('Loss history')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.plot(val_loss, label = 'val')
    ax.legend()
    plt.savefig(os.path.join("/work", "exam", "ASD_classification", "out", "loss_curve_" + classifier + ".png"))

    return model, n_features


###### EVALUATING THE MODEL ######
def Test(n_features, model, x_test, y_test, classifier):
    """ Function to evaluate model on test set
    Args:
        n_features (int): number of features
    """

    # Write classification report
    predicted = model(x_test).detach().numpy()
    output = classification_report(y_test, 
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
    Type lr for logistic regression or nn for a simple neural network \
    with one hidden layers containing 30 nodes: ")
    print(f'\n Your chosen classifier {classifier} will be fitted shortly')
    # xxx make sure to revisit this text when you have decided on the size of the network

    # # User defined variables
    data_path = "/work/exam/ASD_classification/data/dataframes/data_eigstig_text_label.csv"
    epochs = 7000
    learning_rate = 1e-4 # 1e-4 = 0.0001, 1e-3 = 0.001

    # Splitting into separate datasets
    x_train, y_train, x_val, y_val, x_test, y_test = Vectorize(data_path)

    # training the model
    model, n_features = Train(data_path, epochs, classifier, learning_rate)

    Test(n_features, model, x_test, y_test, classifier)