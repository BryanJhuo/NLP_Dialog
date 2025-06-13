
import os
from transformers import pipeline
from .call_llm_api import LLMAnalyzer

API_KEY = ""
MODEL_PATH = "./model"

class ModeBPredictor:
    def __init__(self, emotion_model_path, risk_model_path, API_KEY=API_KEY):
        """
        Initializes the ModeBPredictor with paths to the emotion and risk models.
        Args:
            emotion_model_path (str): Path to the emotion classification model.
            risk_model_path (str): Path to the risk classification model.
        Raises:
            FileNotFoundError: If the provided model paths do not exist.
        """

        # Check if the model paths exist
        if not os.path.exists(emotion_model_path):
            raise FileNotFoundError(f"Emotion model path '{emotion_model_path}' does not exist.")
        
        if not os.path.exists(risk_model_path):
            raise FileNotFoundError(f"Risk model path '{risk_model_path}' does not exist.")

        self.emotion_pipe = pipeline("text-classification", model=emotion_model_path, tokenizer=emotion_model_path, return_all_scores=True)
        self.risk_pipe = pipeline("text-classification", model=risk_model_path, tokenizer=risk_model_path, return_all_scores=True)

        self.emotion_label_map = {
            0: "Other",
            1: "Anger",
            2: "Disgust",
            3: "Fear",
            4: "Happiness",
            5: "Sadness",
            6: "Surprise"
        }

        self.risk_label_map = {
            0: "none",
            1: "perfunctory",
            2: "apathetic",
            3: "avoidant"
        }
        # Initialize the LLM analyzer
        self.llm_analyzer = LLMAnalyzer(API_KEY)
        return 
    
    def predict_emotion(self, utterance: str):
        """
        Predicts the emotion of a given utterance.
        Args:
            utterance (str): The input text for emotion prediction.
        Returns:
            dict: A dictionary containing the predicted emotion label, score, detailed results, and label index.
        Raises:
            ValueError: If the utterance is empty.
        """

        # Validate the input utterance
        if not utterance:
            raise ValueError("Utterance cannot be empty.")
        
        # Use the emotion pipeline to predict the emotion
        emo_result = self.emotion_pipe(utterance)[0]
        emo_idx = int(max(emo_result, key=lambda x: x['score'])['label'].split('_')[-1])
        emo_score = max(emo_result, key=lambda x: x['score'])['score']
        emo_label = self.emotion_label_map[emo_idx]

        return {
            "label": emo_label,
            "score": emo_score,
            "detail": emo_result,
            "label_idx": emo_idx
        }
    
    def predict_risk(self, utterance: str):
        """
        Predicts the risk level of a given utterance.
        Args:
            utterance (str): The input text for risk prediction.
        Returns:
            dict: A dictionary containing the predicted risk label, score, detailed results, and label index.
        Raises:
            ValueError: If the utterance is empty.
        """

        # Validate the input utterance
        if not utterance:
            raise ValueError("Utterance cannot be empty.")

        # Use the risk pipeline to predict the risk
        risk_result = self.risk_pipe(utterance)[0]
        risk_idx = int(max(risk_result, key=lambda x: x['score'])['label'].split('_')[-1])
        risk_score = max(risk_result, key=lambda x: x['score'])['score']
        risk_label = self.risk_label_map[risk_idx]

        return {
            "label": risk_label,
            "score": risk_score,
            "detail": risk_result,
            "label_idx": risk_idx
        }
    
    def predict(self, utterance: str):
        emo_pred = self.predict_emotion(utterance)
        risk_pred = None
        
        if emo_pred['label_idx'] == 0: # 'Other' emotion
            risk_pred = self.predict_risk(utterance)    
        
        return {
            "utterance": utterance,
            "emotion": emo_pred['label'],
            "emotion_score": emo_pred['score'],
            "emotion_detail": emo_pred['detail'],
            "risk": risk_pred['label'] if risk_pred else None,
            "risk_score": risk_pred['score'] if risk_pred else None,
            "risk_detail": risk_pred['detail'] if risk_pred else None,
        }
    
    def call_llm_sugestions(self, dialog_results):
        """
        Calls the LLM API to get suggestions based on the dialog results.
        Args:
            dialog_results (list): List of dictionaries containing dialog results.
        Returns:
            str: Suggestions from the LLM.
        Raises:
            ValueError: If dialog_results is empty or not a list.
        """

        if not isinstance(dialog_results, list) or not dialog_results:
            raise ValueError("dialog_results must be a non-empty list.")

        prompt = self.llm_analyzer.compose_prompt(dialog_results)
        return self.llm_analyzer.call_openrouter(prompt)


if __name__ == "__main__":
    print("This is Mode_B's main script.")

    predictor = ModeBPredictor(
        emotion_model_path=f"{MODEL_PATH}/emotion_bert",
        risk_model_path=f"{MODEL_PATH}/risk_bert",
        API_KEY=API_KEY
    )

    utt = "No, I am okay, really. really."

    result = predictor.predict(utt)

    print("==== Prediction Result ====")
    print(result)

    # Call LLM for suggestions
    dialog_results = [result]
    suggestions = predictor.llm_analyzer.analyze(dialog_results)
    print("==== LLM Suggestions ====")
    print(suggestions)

    print("Done.")