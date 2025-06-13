
import os
import sys
import torch
from transformers import BertTokenizer, BertConfig, BertModel, BertPreTrainedModel
import torch.nn as nn

MODEL_PATH = "./model"

class BertForMultiTask(BertPreTrainedModel):
    def __init__(self, config, num_act_labels=4, num_emotion_labels=7):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier_act = nn.Linear(config.hidden_size, num_act_labels)
        self.classifier_emotion = nn.Linear(config.hidden_size, num_emotion_labels)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, labels_act=None, labels_emotion=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(outputs.pooler_output)
        logits_act = self.classifier_act(pooled_output)
        logits_emotion = self.classifier_emotion(pooled_output)

        loss = None
        if labels_act is not None and labels_emotion is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss_act = loss_fct(logits_act, labels_act)
            loss_emotion = loss_fct(logits_emotion, labels_emotion)
            loss = loss_act + loss_emotion

        return {
            'loss': loss,
            'logits_act': logits_act,
            'logits_emotion': logits_emotion
        }

class ModeAPredictor:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path '{model_path}' does not exist.")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.config = BertConfig.from_pretrained(model_path)
        self.model = BertForMultiTask.from_pretrained(model_path, config=self.config)
        self.model.eval() # Set model to evaluation mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.emotion_label_map = {
            0: "Other",
            1: "Anger",
            2: "Disgust",
            3: "Fear",
            4: "Happiness",
            5: "Sadness",
            6: "Surprise"
        }

        self.act_label_map = {
            1: "Inform",
            2: "Question",
            3: "Directive",
            4: "Commissive"
        }

    def infer(self, utterance: str):
        """
        Perform inference on a given utterance to predict act and emotion.
        Args:
            utterance (str): The input text for which to predict the act and emotion.
        Returns:
            tuple: A tuple containing the predicted act and emotion indices.
        Raises:
            ValueError: If the utterance is empty.
        """
        inputs = self.tokenizer(utterance, return_tensors="pt")
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            pred_act = torch.argmax(outputs['logits_act'], dim=1).item()
            pred_emotion = torch.argmax(outputs['logits_emotion'], dim=1).item()
    
        return pred_act, pred_emotion
    
    def predict(self, utterance: str):
        """
        Predict the act and emotion of a given utterance.
        Args:
            utterance (str): The input text for which to predict the act and emotion.
        Returns:
            dict: A dictionary containing the utterance, predicted act, and predicted emotion.
        Raises:
            ValueError: If the utterance is empty.
        """
        if not utterance:
            raise ValueError("Utterance cannot be empty.")
        
        pred_act, pred_emotion = self.infer(utterance)
        return {
            "text": utterance,
            "predicted_act": self.act_label_map.get(pred_act, "Unknown"),
            "predicted_emotion": self.emotion_label_map.get(pred_emotion, "Unknown")
        }


if __name__ == "__main__":
    predictor = ModeAPredictor(f"{MODEL_PATH}/multitask_bert")

    utterance = "Hello, how can I help you today?"
    outputs = predictor.predict(utterance)
    print(outputs)


    # Test a dialogue
    dialogue = [
        "Hello, how can I help you today?",
        "I am looking for a new phone.",
        "What features are you looking for?",
        "I need a good camera and long battery life.",
        "We have several options that fit your needs."
    ]

    for utterance in dialogue:
        outputs = predictor.predict(utterance)
        print(outputs)
        