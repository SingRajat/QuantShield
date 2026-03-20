import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backend.src.ingestion.ocr_engine import OCREngine
from backend.src.ingestion.layout_parser import LayoutParser

def test_ingestion_pipeline(file_path: str):
    print(f"Testing ingestion pipeline on: {file_path}")
    
    # 1. OCR Engine
    engine = OCREngine()
    pages = engine.extract_from_file(file_path)
    
    print("\n--- OCR Output Summary ---")
    for p in pages:
        text_preview = p['text'][:100].replace('\n', ' ') + "..." if p['text'] else "NO TEXT"
        print(f"Page {p['page']} | Method: {p['method']} | Text snippet: {text_preview}")
        
    # 2. Layout Parser
    parser = LayoutParser()
    extracted_sections = parser.extract_holdings_sections(pages)
    
    print("\n--- Layout Parser Extracted Sections ---")
    print(extracted_sections[:500] + "..." if len(extracted_sections) > 500 else extracted_sections)
    print("\n--- End of Output ---")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_ingestion.py <path_to_pdf_or_image>")
        sys.exit(1)
        
    target_file = sys.argv[1]
    test_ingestion_pipeline(target_file)
