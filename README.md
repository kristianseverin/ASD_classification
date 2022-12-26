# ASD_classification
This repository contains data, scripts, and outputs for binary classification of transcripts from child-researcher interactions from children with Autism Spectrum Disorder (ASD) and Typically Developing (TD) children. 

The experiments conducted in this project count logistic regression models, simple neural networks, and a finetuned BERT model.

# src
The scripts in the source-folder contain:

A .ipynb file, 'data_cleaning', used to clean the data used for analysis

A .py script that can run logistic regressions and simple neural networks. The model can be run from bash with the following commands: 
        If you want to run a logistic regression:
        bash run_lr_or_nn.sh epochs learning_rate ngram mindf maxdf Vectorizer hidden_layer_size relu_negative_slope
        where:
            - epochs (int) = number of epochs that the models is trained over
            - learning_rate (float) = the learning rate of the model
            - ngram (int) = the upper boundary for the ngram range
            - mindf (float) = the lower cutoff for the document frequency
            - maxdf (float) = the upper cutoff for the document frequency
            - Vectorizer (str) = must be either CountVectorizer or TfidfVectorizer. Write bow for CountVectorizer or tfidf for TfidfVectorizer.
        For example run: bash run_lr_or_nn.sh 300 0.001 2 0.05 0.95 bow

        or if you want to run a neural network:
        bash run_lr_or_nn.sh epochs learning_rate ngram mindf maxdf Vectorizer hidden_layer_size relu_negative_slope
        where:
            - hidden_layer_size (int) = the number of nodes in the hidden layer
            - relu_negative_slope (float) = the negative slope coefficient for the leaky ReLU activation function
        For example run: bash run_lr_or_nn.sh 300 0.001 2 0.05 0.95 bow 32 0.2


Running the sript will write metrics from the model to the 'out' folder.

A .ipynb file, 'hyperparameter_sweep.ipynb', runs a parameter sweep using Weights & Biases: L. Biewald, “Experiment Tracking with Weights and Biases,” Weights & Biases. [Online]. Available: http://wandb.com/. [Accessed: 26/12/2022].
Software available from wandb.com. 

A .ipynb file, 'training_model', trains a finetuned BERT model. The model was trained in a Google Colab environment with a mounted Google Drive. For pragmatic reasons, any user who should wish to rerun the script should therefore also run it in Colab environment that allows the user to load the data and simultaneously run on a GPU. This link to a Google drive folder contains the dataset the model was trained on (data_eigstig_text_label.csv): https://drive.google.com/drive/folders/11YngKVR2QQCn6J7ejmGFonaAxXwpbqVn?usp=share_link. Alternatively the script could be run from a local computer. In that case the user must download the dataset available in the drive-folder and/or in the 'data' folder of this github-repository.

A .ipynb file, 'testing_model' tests the model trained in 'training_model' on a heldout test dataset ('HeldoutTestData.csv'). The code was run in a Colab environment, and for pragmatic reasons the model-files have been put in a Google Drive folder that can be accesed by mounting Google drive (the folder can be acessed via the link above). Alternatively the script can be run from a local computer. In that case the contents of 'ModelFolder' should be downloaded (be aware that it contains two .bin files that are very large and take up a lot of space). 

A .ipynb, 'hyperparameter_sweep', file runs a hyperparameter sweep by utilizing Weights & Biases (see reference above). The data can be found in the linked Google Drive folder (data_eigstig_text_label.csv). 

# Out 
Contains the resulting metrics from running the model scripts.

# data
Contains the data used to train the models on. 
