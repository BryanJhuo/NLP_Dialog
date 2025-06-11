import os
import ast
import pandas as pd
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ====== Load dataset and preprocess for training ======
def prepare_data():
    def tag2label(tags):
        if len(tags) == 0:
            return "none"  # Default label if no tags are present
        return tags[0]  # Assuming the first tag is the primary risk label

    # if train_risk.csv/test_risk.csv/validation_risk.csv already exist, return
    if os.path.exists('./data/train_risk.csv') and os.path.exists('./data/test_risk.csv') and os.path.exists('./data/validation_risk.csv'):
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
    
    train_df['risk_tags'] = train_df['risk_tags'].apply(ast.literal_eval)
    test_df['risk_tags'] = test_df['risk_tags'].apply(ast.literal_eval)
    validation_df['risk_tags'] = validation_df['risk_tags'].apply(ast.literal_eval)

    train_df['risk_label'] = train_df['risk_tags'].apply(tag2label)
    test_df['risk_label'] = test_df['risk_tags'].apply(tag2label)
    validation_df['risk_label'] = validation_df['risk_tags'].apply(tag2label)

    # Store new file to ./data directory
    train_df.to_csv('./data/risk_train.csv', index=False)
    test_df.to_csv('./data/risk_test.csv', index=False)
    validation_df.to_csv('./data/risk_validation.csv', index=False)
    print("Data preparation completed. Files saved to ./data directory.")
    return

def train_bert_risk():
    # Prepare the data
    prepare_data()

    label_map = {"none": 0, "perfunctory": 1, "apathetic":2, "avoidant": 3}
    label_list = ["none", "perfunctory", "apathetic", "avoidant"]

    # Load the dataset
    data_files = {
        "train": "./data/risk_train.csv",
        "test": "./data/risk_test.csv",
        "validation": "./data/risk_validation.csv"
    }
    dataset = load_dataset("csv", data_files=data_files)

    # Load the tokenizer and process labels
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    def tokenize_function(examples):
        result = tokenizer(examples["utterance"], truncation=True, padding="max_length", max_length=64)
        result["labels"] = [label_map[label] for label in examples["risk_label"]]
        return result
    
    encode_dataset = dataset.map(tokenize_function, batched=True)

    # Load the model
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./risk_bert",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1"
    )

    # define compute metrics function
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = np.argmax(pred.predictions, axis=1)
        return {
            'accuracy': accuracy_score(labels, preds),
            'macro_f1': f1_score(labels, preds, average='macro'),
            'micro_f1': f1_score(labels, preds, average='micro')
        }
    
    # Create the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encode_dataset["train"],
        eval_dataset=encode_dataset["validation"],
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )

    # Train the model
    trainer.train()
    print("Training completed.")

    # Evaluate the model from the test dataset
    test_results = trainer.evaluate(encode_dataset["test"])
    print("Test results:", test_results)

    # Other evaluation metrics
    pred_output = trainer.predict(encode_dataset["test"])
    y_true = pred_output.label_ids
    y_pred = pred_output.predictions.argmax(axis=1)
    print("Accuracy: ", accuracy_score(y_true, y_pred))
    print("Classification Report(Risk Model): \n")
    print(classification_report(y_true, y_pred, target_names=label_list, digits=3))

    # Save the model at model directory(if not exists and create it)
    if not os.path.exists("./risk_bert"):
        os.makedirs("./risk_bert")
    model.save_pretrained("./risk_bert")
    tokenizer.save_pretrained("./risk_bert")
    print("Model and tokenizer saved to ./risk_bert directory.")
    print("Training and evaluation completed successfully.")

    return

# ====== inference example =======
def inference_example():
    from transformers import pipeline
    # Load the model and tokenizer from the saved directory
    if not os.path.exists("./risk_bert"):
        print("Model directory does not exist. Please train the model first.")
        return
    
    pip = pipeline("text-classification", model="./risk_bert", tokenizer="./risk_bert", return_all_scores=True)
    sample = "Maybe we could just meet for coffee or something."
    result = pip(sample)
    print(f"Input Text: {sample}")
    print("Inference Result:")
    for res in result[0]:
        print(res)

    return

# Main flow for training a risk model
def main():
    # train the risk model
    # train_bert_risk()
    # print("Risk model training completed.")

    # inference_example()
    return

main()