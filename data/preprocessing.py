import pandas as pd
import os
import re

# === Create risk words list ===
risk_wordlist = {
    # 敷衍 
    "perfunctory": [
        "whatever", "fine", "sure", "ok", "okay", "you decide", "doesn't matter",
        "as you wish", "up to you", "alright then", "if you say so", "that's fine",
        "i guess so", "yeah yeah", "mm-hmm"
    ],
    # 冷漠
    "apathetic": [
        "i don't care", "not really", "meh", "nothing much", "so what", "idk",
        "whatever you say", "nothing in particular", "doesn't bother me", "whatever works",
        "i suppose", "it is what it is", "i’m not sure", "i feel nothing"
    ],
    # 逃避
    "avoidant": [
        "let's talk later", "i'm busy", "maybe", "not now", "we’ll see",
        "i don’t know", "never mind", "forget it", "change the topic",
        "anyway", "can we not?", "it's complicated", "i’ll tell you later",
        "don't worry about it", "just drop it"
    ]
}

def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    df = pd.read_csv(file_path, encoding='utf-8')
    if df.empty:
        raise ValueError("The DataFrame is empty. Please check the input file.")
    return df

def risk_detect(sentence, risk_wordlist, threshold=2.0):
    risk_score = 0.0
    risk_tags = []
    sentence_lc = sentence.lower()
    # 用 set 防重複計分
    detected_types = set()
    for risk_type, keywords in risk_wordlist.items():
        sub_score = 0
        for w in keywords:
            # 比對完整詞/短語（不只是 in，避免 "ok" 對到 "joke"）
            if re.search(r'\b' + re.escape(w) + r'\b', sentence_lc):
                sub_score += 1
        if sub_score > 0:
            risk_tags.append(risk_type)
            detected_types.add(risk_type)
            risk_score += sub_score
    # 補充短句加權（可微調）
    if len(sentence_lc.split()) <= 4:
        risk_score += 0.5
    risk_flag = risk_score >= threshold
    return risk_score, risk_tags, risk_flag


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    output_rows = []
    for idx, row in df.iterrows():
        # 處理 acts/emotions：空白分割成 int list
        acts = [int(x) for x in str(row['acts']).strip().split()]
        emotions = [int(x) for x in str(row['emotions']).strip().split()]
        # utterances 以 __eou__ 切分，並去除頭尾空白和多餘空字串
        utterances = [u.strip() for u in str(row['utterances']).split('__eou__') if u.strip()]
        # 逐句分析
        for i, sent in enumerate(utterances):
            risk_score, risk_tags, risk_flag = risk_detect(sent, risk_wordlist)
            output_rows.append({
                "dialogue_id": idx,
                "utterance_id": i,
                "utterance": sent,
                "act": acts[i] if i < len(acts) else None,
                "emotion": emotions[i] if i < len(emotions) else None,
                "risk_score": risk_score,
                "risk_tags": risk_tags,
                "risk_flag": risk_flag
            })
    
    output_df = pd.DataFrame(output_rows)
    output_df = output_df[["dialogue_id", "utterance_id", "utterance", "act", "emotion", "risk_score", "risk_tags", "risk_flag"]]
    return output_df

    

def main():
    path_list = ["train.csv", "test.csv", "validation.csv"]

    # Load and process each file.
    for path in path_list:
        file_path = os.path.join("data", path)
        try:
            df = load_data(file_path)
            print(f"Loaded {path} successfully with {len(df)} records.")
        except (FileNotFoundError, ValueError) as e:
            print(e)
            continue
        
        # Preprocess the DataFrame
        processed_df = preprocess(df)
        output_path = os.path.join("data", f"processed_{path}")
        processed_df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Processed {path} and saved to {output_path} with {len(processed_df)} records.")

    # Test the preprocessd data
    test_file = os.path.join("data", "processed_test.csv")
    if os.path.exists(test_file):
        test_df = pd.read_csv(test_file, encoding='utf-8')
        print(f"Test data loaded successfully with {len(test_df)} records.")
        # Display the first few rows of the test data
        print(test_df.head())
    else:
        print(f"Test data file {test_file} does not exist.")

if __name__ == "__main__":
    main()

        