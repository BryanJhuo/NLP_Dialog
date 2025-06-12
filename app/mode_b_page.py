
import streamlit as st
import sys
import os
# Ensure the parent directory is in the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Mode_B import ModeBPredictor


def mode_b_page():
    if "modeb" not in st.session_state:
        st.session_state["modeb"] = ModeBPredictor(
            emotion_model_path="./model/emotion_bert",
            risk_model_path="./model/risk_bert"
        )

    st.title("NLP Dialog Analysis - Mode B")
    st.write("請在下方輸入一段或多句英文對話，每一行代表一句話，然後點擊「分析」按鈕。")

    # Multi-line text input for user utterances
    user_input = st.text_area("請貼上對話紀錄(每行一句話):", height=200)
    analyze = st.button("分析", type="primary") and user_input.strip()

    if analyze:
        utterances = [line.strip() for line in user_input.split('\n') if line.strip()]
        if not utterances:
            st.error("請至少輸入一句話。")
            return
        results = [st.session_state["modeb"].predict(utt) for utt in utterances]
        st.session_state["sentence_results"] = results

        # LLM suggestions
        st.session_state["llm_result"] = st.session_state["modeb"].call_llm_sugestions(results)
        st.success("分析完成！請查看下方結果。")
        
    mode = st.radio("顯示模式", ("逐句模型分析", "LLM 綜合建議"))   
    if mode == "逐句模型分析":
        # Display results in a table format
        results = st.session_state.get("sentence_results", None)
        if not results:
                st.write("沒有可顯示的結果。請確保輸入的對話紀錄格式正確。")
        else:
            for i, res in enumerate(results):
                st.markdown(f"**第 {i+1} 句** `{res['utterance']}`")
                st.markdown(f"情緒預測： :blue[{res['emotion']}] （分數: {res['emotion_score']:.3f}）")
                if res['emotion'] == "Other":
                    st.markdown(f"風險預測：:red[{res['risk']}]（分數: {res['risk_score']:.3f}）" if res["risk"] != "none" else "風險預測：:green[無明顯風險]")
                st.divider()
        
    elif mode == "LLM 綜合建議":
        llm_result = st.session_state.get("llm_result", None)
        if llm_result:
            st.markdown(llm_result)
        else:
            st.info("尚未生成 LLM 綜合建議。請先進行逐句模型分析。")
    
    else:
        st.info("請輸入對話紀錄並點擊「分析」按鈕。")
    

    st.caption("@ 2024 NLP Dialog Analysis Team. Powered by Streamlit.")
    return 