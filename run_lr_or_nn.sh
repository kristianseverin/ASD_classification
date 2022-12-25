# Install packages
python3 -m pip install --upgrade pip
python3 -m pip install datasets
python3 -m pip install torch
python3 -m pip install sklearn
python3 -m pip install numpy

# Run the code
python3 src/lr_nn_classifier.py --epochs $1 --learning_rate $2 --ngram $3 --mindf $4 --maxdf $5 --Vectorizer $6 --hidden_layer_size $7 --relu_negative_slope $8 # arguments from argparse
# The vectorizer can be either bow (CountVectorizer) or tfidf (TfidfVectorizer)


# To run it (example args)
    ## To run a logistic regression, you need only insert values for learning rate, epochs, ngram, and xxx (max/min_df)
        # bash run_lr_or_nn.sh 300 0.001 2 0.05 0.95 bow

    ## To run a neural network, you must enter values for number of nodes in the hidden layer and for the negative slope coefficient of the leaky ReLU activation function.
        # bash run_lr_or_nn.sh 300 0.001 2 0.05 0.95 bow 32 0.2