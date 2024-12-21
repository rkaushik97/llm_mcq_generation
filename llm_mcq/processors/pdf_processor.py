from pathlib import Path
from typing import List, Dict
import fitz
from tqdm import tqdm

class PDFProcessor:
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)

    def extract_text(self) -> List[Dict[str, str]]:
        pdf_paths = list(self.base_path.rglob("*.pdf"))
        documents = []

        for pdf_path in tqdm(pdf_paths, desc="Processing PDFs"):
            try:
                with fitz.open(pdf_path) as doc:
                    for page_number in range(len(doc)):
                        page = doc[page_number]
                        extracted_text = page.get_text()

                        documents.append({
                            "file": str(pdf_path.relative_to(self.base_path)),
                            "page": page_number + 1,
                            "text": extracted_text
                        })
            except Exception as e:
                print(f"Error processing {pdf_path}: {e}")

        return documents

"""
if __name__ == "__main__":
    base_path = "/Users/apple/unibe_fall/NLP/llm_mcq_generation/intelliprocure_data/"
    processor = PDFProcessor(base_path)
    extracted_documents = processor.extract_text()

    # Print summary of extraction
    print(f"Extracted text from {len(extracted_documents)} pages.")
    for doc in extracted_documents[:5]:
        print(doc)
"""