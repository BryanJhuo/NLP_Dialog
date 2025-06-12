import requests

class LLMAnalyzer:
    def __init__(self, api_key: str):
        self.api_key = api_key
        if not self.api_key:
            raise ValueError("API key must be provided.")
        
        self.url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = "meta-llama/llama-3.3-70b-instruct:free"
        self.max_tokens = 800
        self.temperature = 0.6

    def compose_prompt(self, dialog_results):
        table = "| # | Utterance | Emotion | Score | Risk | RiskScore |\n"
        table += "|---|-----------|---------|-------|------|----------|\n"

        for i, r in enumerate(dialog_results):
            table += f"| {i+1} | {r['utterance']} | {r['emotion']} | {r['emotion_score']:.2f} | {r.get('risk', 'none')} | {r.get('risk_score', '-') if r.get('risk_score') is not None else '-'} |\n"

        task = (
            "你是一位善於理解人際溝通的 NLP 專家。請根據下列表格內容：\n"
            "1. 分析這段對話的整體情緒與風險（如冷漠、敷衍、逃避等）。\n"
            "2. 若發現有明顯問題，請指出具體句子與理由。\n"
            "3. 請給我三個改善溝通的建議回覆。\n"
            "請用繁體中文、條列式回覆。\n\n"
            f"【對話分析表格】\n{table}\n"
        )
        return task

    def call_openrouter(self, prompt):
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.model,
            "messages": [
                {"role": "system","content": "You are an expert in NLP and communication analysis."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=60)
        response.raise_for_status()  # Raise an error for bad responses
        result = response.json()
        return result['choices'][0]['message']['content']

    def analyze(self, dialog_results):
        """
        Analyzes the dialog results by composing a prompt and calling the OpenRouter API.
        Args:
            dialog_results (list): A list of dictionaries containing dialog results with keys 'utterance', 'emotion', 'emotion_score', 'risk', and 'risk_score'.
        Returns:
            str: The response from the OpenRouter API.
        """
        prompt = self.compose_prompt(dialog_results)
        return self.call_openrouter(prompt)
