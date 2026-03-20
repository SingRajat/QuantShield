import os
import tempfile
import fitz  # PyMuPDF

class DocumentProcessor:
    """
    Handles ingestion of various document types (PDFs, images),
    standardizing them for the OCR Engine.
    """
    
    @staticmethod
    async def save_upload_to_temp(upload_file) -> str:
        """Saves an uploaded FastAPI file to a temporary file."""
        suffix = os.path.splitext(upload_file.filename)[1]
        fd, temp_path = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(fd, 'wb') as f:
            content = await upload_file.read()
            f.write(content)
        return temp_path

    @staticmethod
    def preprocess_image(image_path: str):
        """
        Basic image preprocessing for OCR:
        - Grayscale
        - Deskewing (optional, simple logic)
        - Denoising
        """
        import cv2
        import numpy as np
        
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image at {image_path}")
            
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray, h=30)
        
        return denoised

    @staticmethod
    def pdf_page_to_image(page: fitz.Page, dpi: int = 200):
        """Converts a PyMuPDF page to a numpy array image for PaddleOCR."""
        import cv2
        import numpy as np
        
        pix = page.get_pixmap(dpi=dpi)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        
        if pix.n == 4:  # Has alpha channel
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        elif pix.n == 1: # Grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            
        return img
