import logging
from backend.src.ingestion.ocr_engine import OCREngine
from backend.src.ingestion.layout_parser import LayoutParser
from backend.src.ingestion.llm_extractor import LLMExtractor
from backend.src.ingestion.validator import Validator

logger = logging.getLogger(__name__)

class ETFIngestionPipeline:
    """
    Facade tying the entire ETF Document extraction process together.
    """
    def __init__(self, llm):
        self.ocr_engine = OCREngine()
        self.layout_parser = LayoutParser()
        self.llm_extractor = LLMExtractor(llm=llm)
        self.validator = Validator()

    def pre_clean_text(self, text: str) -> str:
        """CRITICAL PRE-CLEANING (Filter out noisy rows)"""
        lines = text.split("\n")
        filtered = []
        for line in lines:
            if "Cash" in line or "USD" in line:
                continue
            if "%" not in line:
                continue
            filtered.append(line)
        return "\n".join(filtered)

    def process_file(self, file_path: str) -> dict:
        """End-to-End processing of an ETF upload."""
        logger.info(f"Processing {file_path}...")
        
        # 1. OCR (PyMuPDF -> PaddleOCR)
        pages = self.ocr_engine.extract_from_file(file_path)
        
        # 2. Layout Parser
        extracted_text = self.layout_parser.extract_holdings_sections(pages)
        
        # 3. Clean Text
        cleaned_text = self.pre_clean_text(extracted_text)
        
        # 4. LLM Extraction
        extraction_result = self.llm_extractor.extract(cleaned_text)
        holdings = extraction_result.get("holdings", [])
        
        # 5. Validate & Deduplicate
        holdings = self.validator.deduplicate(holdings)
        validation = self.validator.validate_weights(holdings)
        
        return {
            "holdings": holdings,
            "status": validation["status"],
            "total_weight": validation["total_weight"]
        }
