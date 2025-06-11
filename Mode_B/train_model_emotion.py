# This file is part of the Mode-B project.
# 
import os
import pandas as pd
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np

# ====== Load dataset and preprocess for training ======
def prepare_data():
    """
    Load the original dataset from the ./data directory and preprocess it for training.
    This function reads the processed_train.csv, processed_test.csv, and processed_validation.csv files,
    extracts the "utterance" and "emotion" columns, converts the emotion labels to integers, and saves
    the processed data to new CSV files in the ./data directory.
    The new files are named train_emo.csv, test_emo.csv, and validation_emo.csv.
    The function does not return any value but prints a message indicating the completion of data preparation.
    It handles file not found errors gracefully and informs the user if the data files are missing.
    """
    # if train_emo.csv/test_emo.csv/validation_emo.csv already exist, return
    if os.path.exists('./data/train_emo.csv') and os.path.exists('./data/test_emo.csv') and os.path.exists('./data/validation_emo.csv'):
        print("Data files already exist. Skipping preparation.")
        return

    # load origin data from ./data directory (processed_train.csv/processed_test.csv/processed_validation.csv)
    try:
        train_df = pd.read_csv('./data/processed_train.csv')
        test_df = pd.read_csv('./data/processed_test.csv')
        validation_df = pd.read_csv('./data/processed_validation.csv')
    except FileNotFoundError as e:
        print(f"Error loading data files: {e}")
        return None
    
    # Only keep "utterance" and "emotion" columns
    train_df = train_df[['utterance', 'emotion']]
    test_df = test_df[['utterance', 'emotion']]
    validation_df = validation_df[['utterance', 'emotion']]

    # Type conversion
    train_df['emotion'] = train_df['emotion'].astype(int)
    test_df['emotion'] = test_df['emotion'].astype(int)
    validation_df['emotion'] = validation_df['emotion'].astype(int)

    # Store new file to ./data directory
    train_df.to_csv('./data/train_emo.csv', index=False)
    test_df.to_csv('./data/test_emo.csv', index=False)
    validation_df.to_csv('./data/validation_emo.csv', index=False)
    print("Data preparation completed. Files saved to ./data directory.")
    return 


# ====== Use Hugging Face datasets to load the dataset ======
def load_data():
    """
    Load the dataset from the ./data directory using Hugging Face's datasets library.
    This function attempts to load the train, validation, and test datasets from CSV files located in the
    ./data directory. If successful, it returns a DatasetDict containing the train, validation, and test datasets.
    If there is an error during loading (e.g., files not found), it prints an error message and returns None.
    Returns:
        DatasetDict: A dictionary containing the train, validation, and test datasets if loading is successful.
        None: If there is an error during loading.
    Raises:
        Exception: If there is an error loading the dataset, it prints the error message.
    """
    try:
        data_files = {
            "train": "./data/train_emo.csv",
            "validation": "./data/validation_emo.csv",
            "test": "./data/test_emo.csv"
        }
        dataset = load_dataset('csv', data_files=data_files)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
    
    return dataset

# ====== Tokenizer and process ======
def tokenize_dataset(dataset, model_ckpt="bert-base-uncased", max_len= 64):
    """
    Tokenize the dataset using the specified model checkpoint and maximum sequence length.
    This function uses the AutoTokenizer from the Hugging Face Transformers library to tokenize the 'utterance' field
    in the dataset. It applies padding and truncation to ensure that all sequences are of the same length.
    Args:
        dataset (DatasetDict): The dataset to be tokenized, containing 'train', 'validation', and 'test' splits.
        model_ckpt (str): The model checkpoint to use for tokenization. Default is "bert-base-uncased".
        max_len (int): The maximum sequence length for tokenization. Default is 64.
    Returns:
        DatasetDict: The tokenized dataset with 'input_ids', 'attention_mask', and 'emotion' fields.
        AutoTokenizer: The tokenizer used for tokenization.
    """
    if dataset is None:
        print("Dataset is None. Cannot tokenize.")
        return None, None
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    def tokenize_function(examples):
        # encode the 'utterance' field
        result = tokenizer(examples['utterance'],padding='max_length',truncation=True,max_length=max_len)
        # add "labels" for model calculates loss
        result['labels'] = examples['emotion']
        return result
        
    encoded_dataset = dataset.map(tokenize_function, batched=True)
    return encoded_dataset, tokenizer

# ====== Compute metrics for evaluation ======
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'macro_f1': f1_score(labels, preds, average='macro'),
        'micro_f1': f1_score(labels, preds, average='micro'),
    }

# ====== main train flow ======
def train_bert_emotion():
    model_ckpt = "bert-base-uncased"
    num_labels = 7 # Assuming 7 emotion classes
    max_len = 64 # Maximum sequence length

    # Step1 : Load and prepare the dataset
    prepare_data()

    # Step2 : Load the dataset
    dataset = load_data()
    if dataset is None:
        print("Dataset loading failed.")
        return
    
    # Step3 : Tokenize the dataset
    encoded_dataset, tokenizer = tokenize_dataset(dataset, model_ckpt, max_len)

    # Step4 : Load the model
    model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=num_labels)
    if model is None:
        print("Model loading failed.")
        return
    
    # Step5 : Define training arguments
    training_args = TrainingArguments(
        output_dir="./emotion_bert",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1"
    )

    # Step6 : Trainer
    trainer =  Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset['train'],
        eval_dataset=encoded_dataset['validation'],
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )

    # Step7 : Train the model
    trainer.train()
    print("Training completed.")

    # Step8 : Evaluate the test dataset
    test_metrics = trainer.evaluate(encoded_dataset['test'])
    print("Test Metrics:", test_metrics)

    # Step9 : Other evaluations
    pred_output = trainer.predict(encoded_dataset['test'])
    y_true = pred_output.label_ids
    y_pred = pred_output.predictions.argmax(axis=1)
    print("\nAccuracy:", accuracy_score(y_true, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, digits=3))

    # Step10 : Save the model and tokenizer at model directory(if not exist, create it)
    if not os.path.exists("./emotion_bert"):
        os.makedirs("./emotion_bert")
    model.save_pretrained("./emotion_bert")
    tokenizer.save_pretrained("./emotion_bert")
    print("Model and tokenizer saved to ./emotion_bert directory.")

    print("Training and evaluation completed successfully.")
    return 

# ====== inference example =======
def inference_example():
    from transformers import pipeline
    # Load the model and tokenizer from the saved directory
    if not os.path.exists("./emotion_bert"):
        print("Model directory does not exist. Please train the model first.")
        return
    pipe = pipeline("text-classification", model="./emotion_bert", tokenizer="./emotion_bert", return_all_scores=True)
    sample_text = "No, I am ok, really."
    result = pipe(sample_text)
    print(f"Input Text: {sample_text}")
    print("Inference Result:")
    for res in result[0]:
        print(res)
    
    return 

def main():
    # train_bert_emotion()
    # inference_example()
    return

main()