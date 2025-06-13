import streamlit as st
import sys
import os 
# Ensure the parent directory is in the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Mode_A import ModeAPredictor

def mode_a_page():
    if "modea" not in st.session_state:
        st.session_state["modea"] = ModeAPredictor(model_path="./model/multitask_bert")
    
    st.title("NLP Dialog Analysis - Mode A")
    st.write("請在下方輸入英文對話，每一行代表一句，按下「分析」即可逐句分析。")

    user_input = st.text_area("請貼上對話紀錄(每行一句話):", height=200)

    if st.button("分析", type="primary") and user_input.strip():
        st.subheader("分析結果")
        utterances = [line.strip() for line in user_input.split('\n') if line.strip()]
        if not utterances:
            st.error("請至少輸入一句話。")
            return
        result = []

        for utt in utterances:
            try:
                pred = st.session_state["modea"].predict(utt)
                result.append(pred)
            except Exception as e:
                st.error(f"處理句子時發生錯誤：{str(e)}")
                
        if not result:
            st.write("沒有可顯示的結果。請確保輸入的對話紀錄格式正確。")
            return
        else: 
            for i, res in enumerate(result):
                st.markdown(f"**第 {i+1} 句** `{res['text']}`")
                st.markdown(f"- 語意分類（Act）：:blue[{res['predicted_act']}]")
                st.markdown(f"- 情緒分類（Emotion）：:red[{res['predicted_emotion']}]")
                st.divider()
    else:
        st.info("請輸入對話紀錄並點擊「分析」按鈕。")

    st.caption("@ 2024 NLP Dialog Analysis Team. Powered by Streamlit.")
    return 
