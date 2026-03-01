class MockLLMAgent:
    """
    Provides a deterministic textual explanation of computed risk metrics.
    Satisfies constraints avoiding synthetic data and forecasting - strictly backward-looking.
    """
    def generate_explanation(self, risk_class: str, features: dict) -> str:
        vol = features.get('Annualized_Volatility', 0)
        var = features.get('Historical_VaR_95', 0)
        max_dd = features.get('Maximum_Drawdown', 0)
        div_ratio = features.get('Diversification_Ratio', 0)
        
        return f"Based on its past performance, this portfolio is currently considered {risk_class} risk. " \
               f"On average, its value goes up or down by about {vol:.2%} each year. " \
               f"Looking at bad market days in the past, a typical 'worst-case' daily drop has been around {var:.2%}. " \
               f"The biggest overall drop it ever took from a high point to a low point was {max_dd:.2%}. " \
               f"It has a diversification score of {div_ratio:.2f}—a higher score means your eggs are better spread across different baskets, which helps lower your overall risk."
