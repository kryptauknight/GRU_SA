# GRU Sentiment Analysis Model

## **Overview**

This project implements a Gated Recurrent Unit model using PyTorch for the sentiment analysis task. The GRU model is particularly effective for sequence data like text and has been used here to classify textual data into **sentiment** categories.

## **Model Architecture**

The model uses the following architecture:

- An **Embedding Layer** to convert text data into dense vector representations: 300 dimensional Embeddings ['https://drive.google.com/file/d/1vTBJ0CerCJ_3RQgw7op9MJNC_Fjs7cis/view?usp=drive_link']
- One **GRU Layer** to process the sequence data 
- A **Fully Connected Layer** to interpret the GRU output and make predictions.
- **Dropout** for regularization to prevent overfitting, **Dropout Rate = 0.5**

## **Dataset**

The dataset used for this project consists of 100,000 reviews of hotels in Arabic (All varying in lengths). It was originally stripped of any diacretics from the get go and It is proper arabic and not any specific dialect.

## **Preprocessing**

The text data was preprocessed using the following steps:

1. Tokenization: Splitting text into individual words or tokens.
2. Text Cleaning: Removing special characters, numbers, and other non-essential elements.
3. Vectorization: Converting tokens into numerical format for model input.
4. Padding: Ensuring all sequences are of uniform length. (by adding <PAD> tokens or 0s)

## **Training**

The model was trained using:

- Loss Function: Cross-Entropy Loss, suitable for multi-class classification.
- Optimizer: Adam.
- Number of Epochs: 10 epochs.

## **Requirements**

Key Python libraries used in this project include:

- PyTorch
- Pandas
- NumPy
- Gensim
- NLTK

These can be installed using the **`requirements.txt`** file provided.

## **Usage**

In order to use the model for sentiment analysis:

1. Preprocess your input text data using the same steps as in the training phase.
2. Load the model using the saved state dictionary.
3. Pass the processed data through the model to get predictions.

## **How to Run**

1. Load the model using its state dictionary: ['https://drive.google.com/file/d/1mINRBykdA9Ch6G3cRQvL7mm4g4xgc6E_/view?usp=drive_link']
2. Install the required packages: **`pip install -r requirements.txt`**.
3. Open the **`GRU.ipynb`** notebook in a Jupyter environment.
4. Load the Model's state dictionary.
5. Run the cells in sequence to train and evaluate the model.

## **Contact**

For queries or further information, please contact [zaamountaha@gmail.com].
