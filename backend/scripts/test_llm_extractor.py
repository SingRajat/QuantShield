import sys
import os
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
except ImportError:
    from langchain.chat_models.fake import FakeListChatModel
from backend.src.ingestion.llm_extractor import LLMExtractor

def test_llm_extractor():
    print("Testing LLMExtractor pipeline...")
    
    # 1. Define a fake LLM response (this is what the chat model would output internally before Pydantic parsing)
    # The PydanticOutputParser expects pure JSON or JSON fenced in ```json ... ``` blocks.
    fake_response = json.dumps({
        "holdings": [
            {"name": "Apple Inc.", "weight": 10.5},
            {"name": "Microsoft Corp", "weight": 8.2}
        ]
    })
    
    fake_llm = FakeListChatModel(responses=[fake_response])
    
    # 2. Instantiate extractor
    extractor = LLMExtractor(llm=fake_llm)
    
    # 3. Provide a sample OCR chunk
    sample_text = (
        "Top 10 Holdings\n"
        "Ticker      Company Name                           Weight (%)\n"
        "AAPL        Apple Inc.                             10.5\n"
        "MSFT        Microsoft Corp                         8.2\n"
        "--- Cash and Equivalents ---\n"
        "            USD                                    2.0\n"
    )
    
    # 4. Extract
    print(f"Sample text provided to LLM:\n{sample_text}")
    print("\nExecuting chain (Mocked LLM will return valid JSON)...")
    result = extractor.extract(sample_text)
    
    # 5. Output Verification
    print("\n--- Extraction Result ---")
    print(json.dumps(result, indent=2))
    
    # Assertions
    assert "holdings" in result, "Result is missing 'holdings' key"
    assert isinstance(result["holdings"], list), "'holdings' is not a list"
    assert len(result["holdings"]) == 2, f"Expected 2 holdings, got {len(result['holdings'])}"
    
    h1 = result["holdings"][0]
    assert h1["name"] == "Apple Inc.", f"Expected Apple Inc., got {h1['name']}"
    assert float(h1["weight"]) == 10.5, f"Expected 10.5, got {h1['weight']}"

    print("\nSUCCESS: Output formatting and strict schema enforcement verified.")

if __name__ == "__main__":
    test_llm_extractor()
