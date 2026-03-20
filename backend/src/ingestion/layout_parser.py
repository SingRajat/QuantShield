import logging
import re

logger = logging.getLogger(__name__)

class LayoutParser:
    """
    Parses OCR'd text to identify sections that likely contain ETF holdings.
    """
    
    def extract_holdings_sections(self, pages: list) -> str:
        """
        Takes the output from OCREngine and returns a concatenated string
        of sections most likely to contain holdings (e.g., tables, list of percentages).
        """
        holdings_text = []
        
        for page in pages:
            text = page.get("text", "")
            
            if self._is_likely_holdings_page(text):
                logger.info(f"Page {page['page']} identified as likely containing holdings.")
                holdings_text.append(f"--- PAGE {page['page']} ---\n{text}")
                
        # If heuristics fail to find a clear holdings page, fallback to passing all text
        if not holdings_text:
            logger.warning("No explicit holdings sections detected. Passing all text.")
            for page in pages:
                holdings_text.append(f"--- PAGE {page['page']} ---\n{page.get('text', '')}")
                
        return "\n\n".join(holdings_text)
        
    def _is_likely_holdings_page(self, text: str) -> bool:
        keywords = ["holdings", "weight", "portfolio", "allocation", "ticker", "security", "top 10"]
        text_lower = text.lower()
        
        # Check keyword presence
        keyword_hits = sum(1 for kw in keywords if kw in text_lower)
        
        # Check percentage density
        pct_matches = len(re.findall(r'\d+(\.\d+)?%', text))
        
        return keyword_hits >= 1 or pct_matches >= 3
