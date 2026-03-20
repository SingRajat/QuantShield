import traceback
import sys

try:
    from backend.scripts.test_ingestion import test_ingestion_pipeline
    test_ingestion_pipeline('dummy.pdf')
    print("SUCCESS")
except Exception as e:
    with open('traceback.txt', 'w') as f:
        traceback.print_exc(file=f)
    print("FAILED - check traceback.txt")
