import re
from typing import List, Dict


class TextPreprocessor:
    @staticmethod
    def clean_text(text:str) -> str:
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s.,!?]", "", text)
        return text

    @staticmethod
    def preprocess_documents(documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
        for doc in documents:
            doc["text"] = TextPreprocessor.clean_text(doc["text"])
        return documents
