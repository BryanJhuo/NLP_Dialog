import pandas as pd
import torch
from transformers import BertTokenizer, BertConfig
from train_model import BertForMultiTask, NLPMultiTaskDataset
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

# 設定路徑
CHECKPOINT_PATH = './results_modelA/checkpoint-5449'
TEST_CSV_PATH = 'your-repo/processed_test.csv'
OUTPUT_CSV_PATH = 'your-repo/predicted_test.csv'

# 載入 tokenizer 與模型
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
config = BertConfig.from_pretrained(CHECKPOINT_PATH)
model = BertForMultiTask.from_pretrained(CHECKPOINT_PATH, config=config)
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 讀入測試資料
test_df = pd.read_csv(TEST_CSV_PATH)
test_dataset = NLPMultiTaskDataset(test_df, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=16)

# 預測結果儲存區
all_preds_act, all_preds_emotion = [], []
all_labels_act, all_labels_emotion = [], []

# 預測流程
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels_act = batch['labels_act'].to(device)
        labels_emotion = batch['labels_emotion'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds_act = torch.argmax(outputs['logits_act'], dim=1)
        preds_emotion = torch.argmax(outputs['logits_emotion'], dim=1)

        all_preds_act.extend(preds_act.cpu().numpy())
        all_preds_emotion.extend(preds_emotion.cpu().numpy())
        all_labels_act.extend(labels_act.cpu().numpy())
        all_labels_emotion.extend(labels_emotion.cpu().numpy())

# ====== 分類報告與錯誤分析 ======
print("\n=== ACT 分類報告 ===")
print(classification_report(all_labels_act, all_preds_act, digits=3))
print("\n=== EMOTION 分類報告 ===")
print(classification_report(all_labels_emotion, all_preds_emotion, digits=3))

# 繪製 confusion matrix
def plot_cm(true, pred, title, labels):
    cm = confusion_matrix(true, pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.title(f"Confusion Matrix: {title}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"your-repo/confusion_matrix_{title.lower()}.png")
    plt.close()

plot_cm(all_labels_act, all_preds_act, "ACT", labels=[1,2,3,4])
plot_cm(all_labels_emotion, all_preds_emotion, "EMOTION", labels=list(range(7)))

# 儲存預測結果
test_df['pred_act'] = all_preds_act
test_df['pred_emotion'] = all_preds_emotion
test_df['act_correct'] = test_df['act'] == test_df['pred_act']
test_df['emotion_correct'] = test_df['emotion'] == test_df['pred_emotion']
test_df.to_csv(OUTPUT_CSV_PATH, index=False)

print(f"\n✅ 預測完成，已儲存至：{OUTPUT_CSV_PATH}")
print("📊 混淆矩陣圖已儲存至 your-repo 資料夾。")
