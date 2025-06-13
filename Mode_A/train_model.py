import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertModel, BertPreTrainedModel, TrainingArguments, Trainer
from transformers import BertConfig
import torch.nn as nn
from sklearn.metrics import accuracy_score

# ===== 1. Dataset class =====
class NLPMultiTaskDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        inputs = self.tokenizer(row['utterance'], padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        item = {key: val.squeeze() for key, val in inputs.items()}
        item['labels_act'] = torch.tensor(row['act'], dtype=torch.long)
        item['labels_emotion'] = torch.tensor(row['emotion'], dtype=torch.long)
        return item

# ===== 2. Multi-task Model =====
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

# ===== 3. Metrics =====
def compute_metrics(pred):
    logits_act, logits_emotion = pred.predictions
    labels_act, labels_emotion = pred.label_ids
    preds_act = logits_act.argmax(-1)
    preds_emotion = logits_emotion.argmax(-1)
    return {
        'accuracy_act': accuracy_score(labels_act, preds_act),
        'accuracy_emotion': accuracy_score(labels_emotion, preds_emotion)
    }

# ===== 4. Main training =====
if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    config = BertConfig.from_pretrained('bert-base-uncased')
    model = BertForMultiTask.from_pretrained('bert-base-uncased', config=config)

    # Load data
    train_df = pd.read_csv('your-repo/processed_train.csv')
    val_df = pd.read_csv('your-repo/processed_validation.csv')

    train_dataset = NLPMultiTaskDataset(train_df, tokenizer)
    val_dataset = NLPMultiTaskDataset(val_df, tokenizer)

    training_args = TrainingArguments(
        output_dir='./results_modelA',
        num_train_epochs=1, # 訓練層數ㄥ
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir='./logs',
        load_best_model_at_end=True,
        metric_for_best_model='accuracy_act'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model('./modelA_final')


