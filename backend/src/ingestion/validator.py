import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class Validator:
    """
    Validation Layer (Phase 3)
    Handles Deduplication, Weight checks, and Entity Normalization.
    """
    
    @staticmethod
    def deduplicate(holdings: List[Dict]) -> List[Dict]:
        """Remove duplicate holdings by name."""
        unique = {}
        for h in holdings:
            unique[h["name"]] = h
        return list(unique.values())

    @staticmethod
    def validate_weights(holdings: List[Dict]) -> Dict:
        """Validate if the total weight represents a full or partial ETF."""
        total_weight = sum(h["weight"] for h in holdings)
        
        if total_weight < 50 or total_weight > 110:
            status = "invalid"
            logger.warning(f"Validation failed: Total weight {total_weight}% is outside acceptable bounds.")
        elif total_weight < 95:
            status = "partial"
        else:
            status = "complete"
            
        return {
            "total_weight": total_weight,
            "status": status
        }
