# NLP_Dialog

## How to use or create environment ?

### Fisrt time
1. **安裝必要工具（只需一次）**
    - [Docker Desktop](https://www.docker.com/products/docker-desktop/)
    - [Visual Studio Code](https://code.visualstudio.com/)
    - 安裝以下 VSCode 擴充套件：
        - ✅ Remote - Containers
        - ✅ Python

2. **Clone 專案**
```bash
git clone https://github.com/your-org/your-repo.git
cd your-repo
```

3. **開啟 VSCode 並啟用 Remote Container**
    - 開啟專案資料夾
    - 按下 F1 或 Ctrl+Shift+P
    - 輸入 Remote-Containers: Reopen in Container
    - 等待自動建置並開啟 container（會自動安裝 Python 環境與套件）

## Dataset
#### processed_***.csv
| dialogue_id | utterance_id | utterance | act | emotion | risk_score | risk_tags | risk_flag |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 10 | "Okay , I'll do it next time ." | 4 | 0 | 1.0 | ['perfunctory'] | False |
| 9 | 0 | "Hello , this is Mike , Kara ."| 1 | 0 | 0.0 | [] | False |
| 9 | 6 | "Maybe we could just meet for coffee or something ." | 3 | 0 | 1.0 | ['avoidant']| False |

這是已處理過後的檔案，以下是col的說明：
- dialogue_id: 代表哪一段對話紀錄
- utterance_id: 代表第幾句對話
- utterance: 為每一句對話
- act: 每一個值對應的行為(?)
    - 1: Inform -> 	陳述、提供訊息、解釋、敘述觀點
    - 2: Question -> 提問
    - 3: Directive -> 指令、建議、請求、命令
    - 4: Commissive -> 承諾、答應、主動承擔責任
- emotion: 
    - 0: Other -> 無明確情緒、中性、或無法歸類
    - 1: Anger -> 憤怒、惱怒、激動
    - 2: Disgust -> 厭惡、反感
    - 3: Fear -> 恐懼、害怕
    - 4: Happiness -> 快樂、開心、正向情緒
    - 5: Sadness -> 悲傷、失落
    - 6: Surprise -> 驚訝、出乎意料
- risk_score: 三類別風險分數總和（基礎加總）
- risk_tags: 該句中命中的風險類型
- risk_flag: 若 risk_score >= threshold 則設為 True，預設 threshold = 2.0  
 
風險針對**敷衍**、**冷漠**和**逃避**。所以再後續功能判斷的時候，應該先以 emotions 為主，若 emotion 為 other，才去判斷該句話是否有**敷衍**、**冷漠**和**逃避**的行為，也就是再去看 risk_tag。

#### test/train/validation.csv
此為單純尚未處理的database。如果要用pandas讀入，要注意讀出每個element要去split。
> 不懂可以問我，或者去看原 dataset 的網站。

## References
- DailyDialog: [HuggingFace](https://huggingface.co/datasets/roskoN/dailydialog)