# Sentiment Analysis Project

This file helps to understand all existing Python files.

## Project Structure

The project is organized into the following Python scripts:

-   `prepare_data.py`: Preprocesses the raw IMDB dataset.
-   `Train_machine_learning_model.py`: I tried many models but the best two are Logistic Regression and XGBClassifier model on the preprocessed data.
-   `Used_machine_learning_model.py`: Uses the trained Logistic Regression model or XGBClassifier to predict review.
-   `Train_LSTM_model.py`: Trains an LSTM model on the preprocessed data using two embeddings GloVe and Randomly Initialized.
-   `Train_LSTM_model_with_raw_data.py`: Trains an LSTM model on the raw data using two embeddings GloVe and Randomly Initialized.
-   `Used_LSTM_model.py`: Uses the trained LSTM model to predict review.
-   `GPT-2 Classification.py`:  Performs sentiment classification using a pre-trained GPT-2 model with few-shot prompting.

## Other Files
- I Saved all the model results I trained in json files and named them according to the model.
- I Save the model you trained in the second task so you can use it in the Used_machine_learning_model.py file.
- I saved the weights of the models I trained in the third task to reduce the file size. I can use them through the Used_LSTM_model.py file.
- I saved the tokenizer model used in the training process.

## Notes
- I used Cuda to train Lstm model 
- You need to download the GloVe embeddings, You can download it from the this link [GloVe website](https://nlp.stanford.edu/data/glove.6B.zip)
- I used pre-trained embedding into task two but his results weren't high so I didn't save him.
