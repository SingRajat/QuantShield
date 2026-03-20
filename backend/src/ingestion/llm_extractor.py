import logging
from typing import List, Optional
from pydantic import BaseModel, Field

# We'll use LangChain for structuring the extraction
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
logger = logging.getLogger(__name__)

# Strict Output Schema matching the user's contract
class HoldingItem(BaseModel):
    name: str = Field(description="The canonical name or ticker of the stock extracted from the document.")
    weight: float = Field(description="The percentage weight of the stock in the ETF (e.g., 10.5). Do not include the '%' sign.")

class HoldingsExtraction(BaseModel):
    holdings: List[HoldingItem] = Field(description="List of extracted stock holdings.")

class LLMExtractor:
    """
    RAG-Based Extraction Layer for ETF Holdings.
    Enforces 'Extract, don't infer' rules.
    """
    def __init__(self, llm):
        """
        Expects a LangChain BaseChatModel instance (e.g. ChatOpenAI, ChatGoogleGenerativeAI)
        that supports structured output or good JSON adherence.
        """
        self.llm = llm
        
        # We chunk text if it's extremely long, to fit into context window and improve extraction focus
        # For tables, larger chunks reduce cutting off rows midway.
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=500,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Setup the parser and prompt
        self.parser = PydanticOutputParser(pydantic_object=HoldingsExtraction)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "You are an expert financial data extraction system enforcing a strict 'Extract, don't infer' policy.\n"
             "Your task is to extract only the stock names (or tickers) and their exact percentage weights from the raw OCR text.\n"
             "\nSTRICT RULES:\n"
             "1. Extract ONLY from the text provided. Do NOT hallucinate missing stocks.\n"
             "2. Do NOT guess or estimate missing weights. If a weight is missing or unclear, SKIP the entry entirely.\n"
             "3. Ignore headers, footnotes, disclaimers, and non-holding rows (e.g., 'Cash', 'Total').\n"
             "4. Do NOT attempt to normalize tickers or names. Just extract exactly what you see.\n"
             "5. If there are absolutely no holdings in the text, return an empty 'holdings' list.\n"
             "\n{format_instructions}\n"
            ),
            ("user", "Extract the holdings from the following text:\n\n{text}")
        ])
        
        self.chain = self.prompt | self.llm | self.parser

    def extract(self, text: str) -> dict:
        """
        Extracts holdings from the provided text chunk(s).
        Returns a dictionary representing the extracted holdings.
        """
        if not text or not text.strip():
            logger.warning("Empty text provided to LLMExtractor.")
            return {"holdings": []}

        chunks = self.text_splitter.split_text(text)
        all_holdings = []
        
        for i, chunk in enumerate(chunks):
            logger.info(f"Extracting holdings from chunk {i+1}/{len(chunks)}...")
            try:
                # Run the chain on the chunk
                result = self.chain.invoke({
                    "text": chunk,
                    "format_instructions": self.parser.get_format_instructions()
                })
                
                all_holdings.extend([h.dict() for h in result.holdings])
                
            except Exception as e:
                logger.error(f"Failed to extract from chunk {i+1}: {e}")
                # We optionally fail safely but keep processing other chunks
                continue
                
        return {"holdings": all_holdings}
