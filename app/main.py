import streamlit as st
from mode_a_page import mode_a_page
from mode_b_page import mode_b_page

st.set_page_config(
    page_title="NLP Dialog Analysis",
    page_icon=":speech_balloon:",
    layout="centered"
    )

st.sidebar.title("功能選單")
page = st.sidebar.radio("選擇功能模式", ("主畫面","Mode A：語意分類＋問題抽取", "Mode B：情緒偵測＋風險分析") )

# Main page
if page == "主畫面":
    st.title("NLP Dialog Analysis 主畫面")
    st.write("""
        歡迎使用 NLP Chatlog Analysis。
        
        請從左側選單選擇你要使用的功能模式：
        - **Mode A**：語意分類與問題抽取
        - **Mode B**：情緒偵測與風險分析
    """)
    st.image("https://cdn-icons-png.flaticon.com/512/3649/3649466.png", width=150)  # 任意logo

elif page == "Mode A：語意分類＋問題抽取":
    mode_a_page()

elif page == "Mode B：情緒偵測＋風險分析":
    mode_b_page()