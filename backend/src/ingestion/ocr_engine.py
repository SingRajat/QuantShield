import logging
import re
import sys
import fitz  # PyMuPDF
from backend.src.ingestion.document_processor import DocumentProcessor

logger = logging.getLogger(__name__)

class OCREngine:
    """
    Implements a 2-stage OCR Strategy:
    Stage 1: Native Text Extraction via PyMuPDF (Fast Path).
    Stage 2: OCR Fallback via PaddleOCR for scanned images or noisy PDFs.
    """
    
    def __init__(self):
        # Initialize PaddleOCR lazily to avoid heavy loading/crashes unless needed
        self._paddle_ocr = None

    @property
    def paddle_ocr(self):
        if self._paddle_ocr is None:
            try:
                import sys
                import langchain.docstore.document
            except ImportError:
                from collections import namedtuple
                Document = namedtuple("Document", ["page_content", "metadata"])
                class MockDocstore: pass
                class MockDocstoreDocument: pass
                MockDocstoreDocument.Document = Document
                import langchain
                langchain.docstore = MockDocstore()
                langchain.docstore.document = MockDocstoreDocument()
                sys.modules['langchain.docstore'] = langchain.docstore
                sys.modules['langchain.docstore.document'] = langchain.docstore.document
            
            from paddleocr import PaddleOCR
            self._paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        return self._paddle_ocr

    def extract_from_file(self, file_path: str):
        """Routes the file to the correct extraction strategy based on file type."""
        ext = file_path.lower().split('.')[-1]
        
        if ext == 'pdf':
            return self.extract_from_pdf(file_path)
        elif ext in ['png', 'jpg', 'jpeg']:
            return self.extract_from_image(file_path)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    def extract_from_pdf(self, pdf_path: str):
        """Processes a PDF page by page with the 2-stage strategy."""
        doc = fitz.open(pdf_path)
        extracted_pages = []

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # Stage 1: Native Text Extraction
            native_text = page.get_text()
            
            if self._is_valid_native_text(native_text):
                logger.info(f"Page {page_num + 1}: Using Stage 1 (PyMuPDF native text)")
                extracted_pages.append({
                    "page": page_num + 1,
                    "method": "pymupdf",
                    "text": native_text
                })
            else:
                logger.info(f"Page {page_num + 1}: Using Stage 2 (PaddleOCR fallback)")
                # Stage 2: OCR Fallback
                img = DocumentProcessor.pdf_page_to_image(page, dpi=200)
                ocr_text = self._run_paddle_ocr(img)
                extracted_pages.append({
                    "page": page_num + 1,
                    "method": "paddleocr",
                    "text": ocr_text
                })
        
        doc.close()
        return extracted_pages

    def extract_from_image(self, image_path: str):
        """Processes an image directly using Stage 2 (PaddleOCR)."""
        logger.info(f"Image {image_path}: Using Stage 2 (PaddleOCR)")
        preprocessed_img = DocumentProcessor.preprocess_image(image_path)
        ocr_text = self._run_paddle_ocr(preprocessed_img)
        
        return [{
            "page": 1,
            "method": "paddleocr",
            "text": ocr_text
        }]

    def _run_paddle_ocr(self, img_array) -> str:
        """Executes PaddleOCR on a numpy image array and formats the output."""
        result = self.paddle_ocr.ocr(img_array, cls=True)
        ocr_text = ""
        
        if result and result[0]:
            # result[0] contains a list of lines
            # each line format: [[p1, p2, p3, p4], ('text', confidence)]
            lines = [line[1][0] for line in result[0] if line and len(line) == 2]
            ocr_text = "\n".join(lines)
            
        return ocr_text

    def _is_valid_native_text(self, text: str) -> bool:
        """
        Heuristics to determine if PyMuPDF extracted enough clean text.
        If it's just a scanned PDF, native text will be tiny or empty.
        We also look for numbers/percentages which might indicate holdings tables.
        """
        if not text or len(text.strip()) < 100:
            return False
            
        # Check if there's any percentage sign and digits, hinting at weights.
        if not re.search(r'\d+(\.\d+)?%', text):
            # No percentages found. But it still could be a valid page (e.g., disclaimer). 
            # We'll allow it if text length is very high, but let's be strict.
            if len(text.strip()) < 500:
                return False
                
        return True
