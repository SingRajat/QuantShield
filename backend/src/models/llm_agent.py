import os
import logging
from dotenv import load_dotenv

# Langchain imports
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()
logger = logging.getLogger(__name__)


class LLMAgent:
    """
    Uses LangChain and Groq's Llama 3.3 70B to generate human-readable, beginner-friendly
    explanations of portfolio risk metrics.

    Constraints:
      - Strictly backward-looking (no forecasting or investment advice).
      - Explains only computed historical metrics.
      - Falls back to a deterministic template if the API call fails or key is missing.
    """

    MODEL = "llama-3.3-70b-versatile"

    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.chain = None
        self.llm = None
        
        if self.api_key:
            try:
                # Initialize ChatGroq LLM
                llm = ChatGroq(
                    temperature=0.3, 
                    groq_api_key=self.api_key, 
                    model_name=self.MODEL,
                    max_tokens=300
                )
                
                # Define Prompt Template
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are a financial risk analyst. You explain portfolio risk metrics in simple, beginner-friendly English. You never give investment advice or predictions. You only describe historical facts."),
                    ("user", """The ML model has classified this portfolio as: **{risk_class} Risk**

Here are the computed historical metrics:
- Annualized Volatility: {vol:.2%}
- Historical VaR (95%): {var:.2%}
- Maximum Drawdown: {max_dd:.2%}
- Diversification Ratio: {div_ratio:.2f}
- Skewness: {skewness:.2f}
- Kurtosis: {kurtosis:.2f}
- Sharpe Ratio: {sharpe:.2f}
- Sortino Ratio: {sortino:.2f}
- Beta: {beta:.2f}

Rules you MUST follow:
1. Use simple, plain English. No jargon.
2. Explain what each metric means in beginner-friendly terms.
3. Keep it strictly backward-looking — only describe what HAS happened historically.
4. Do NOT give any investment advice, predictions, or forecasts.
5. Do NOT say "you should" or "I recommend."
6. Keep the response under 150 words.
7. Do NOT use bullet points or markdown formatting — write in plain paragraph form.""")
                ])
                
                # Construct LCEL (LangChain Expression Language) Chain
                self.chain = prompt | llm | StrOutputParser()
                self.llm = llm
                logger.info("LangChain Groq client initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize LangChain Groq client: {e}")
                self.chain = None
        else:
            logger.warning("GROQ_API_KEY not found in .env. LLM agent will use deterministic fallback.")

    def _fallback_explanation(self, risk_class: str, features: dict) -> str:
        """Deterministic fallback if Groq API is unavailable."""
        vol = features.get('Annualized_Volatility', 0)
        var = features.get('Historical_VaR_95', 0)
        max_dd = features.get('Maximum_Drawdown', 0)
        div_ratio = features.get('Diversification_Ratio', 0)

        return (
            f"Based on its past performance, this portfolio is currently considered {risk_class} risk. "
            f"On average, its value goes up or down by about {vol:.2%} each year. "
            f"Looking at bad market days in the past, a typical 'worst-case' daily drop has been around {var:.2%}. "
            f"The biggest overall drop it ever took from a high point to a low point was {max_dd:.2%}. "
            f"It has a diversification score of {div_ratio:.2f} — a higher score means your investments are "
            f"better spread across different areas, which helps lower your overall risk."
        )

    def generate_explanation(self, risk_class: str, features: dict) -> str:
        """
        Generates a human-readable explanation using Llama 3.3 70B via LangChain+Groq.
        Falls back to deterministic template if the API is unavailable.
        """
        if not self.chain:
            logger.info("Using deterministic fallback (no LangChain Groq client).")
            return self._fallback_explanation(risk_class, features)

        try:
            # Extract features for the LangChain formatting
            vol = features.get('Annualized_Volatility', 0)
            var = features.get('Historical_VaR_95', 0)
            max_dd = features.get('Maximum_Drawdown', 0)
            div_ratio = features.get('Diversification_Ratio', 0)
            skewness = features.get('Skewness', 0)
            kurtosis = features.get('Kurtosis', 0)
            sharpe = features.get('Sharpe', 0)
            sortino = features.get('Sortino', 0)
            beta = features.get('Beta', 0)

            # Invoke LangChain pipeline
            explanation = self.chain.invoke({
                "risk_class": risk_class,
                "vol": vol,
                "var": var,
                "max_dd": max_dd,
                "div_ratio": div_ratio,
                "skewness": skewness,
                "kurtosis": kurtosis,
                "sharpe": sharpe,
                "sortino": sortino,
                "beta": beta
            })
            
            logger.info("LLM explanation generated successfully via LangChain Groq.")
            return explanation.strip()

        except Exception as e:
            logger.error(f"LangChain Groq API call failed: {e}. Using deterministic fallback.")
            return self._fallback_explanation(risk_class, features)

    def _fallback_dashboard_explanations(self, risk_class: str, features: dict, portfolio_return: float, benchmark_return: float) -> dict:
        """Deterministic fallback for per-section dashboard insights."""
        vol = features.get('Annualized_Volatility', 0)
        var = features.get('Historical_VaR_95', 0)
        max_dd = features.get('Maximum_Drawdown', 0)
        sharpe = features.get('Sharpe', 0)

        perf_diff = portfolio_return - benchmark_return
        direction = "outperformed" if perf_diff > 0 else "underperformed"

        return {
            "risk_gauge": (
                f"This portfolio has been classified as {risk_class} risk. "
                f"Its annualized volatility is {vol:.2%} and the worst-case daily loss (95% confidence) is {var:.2%}."
            ),
            "advanced_analytics": (
                f"Over the last 6 months, the portfolio has {direction} the benchmark by {abs(perf_diff):.2%}. "
                f"The portfolio's Sharpe ratio is {sharpe:.2f}, indicating its historical risk-adjusted return."
            ),
            "asset_correlation": (
                "The correlation heatmap shows how individual assets moved relative to each other. "
                "Higher correlations (closer to 1.0) mean assets tend to move together, reducing diversification benefit."
            )
        }

    def generate_dashboard_explanations(self, risk_class: str, features: dict, portfolio_return: float, benchmark_return: float) -> dict:
        """
        Generates per-section AI insights for the dashboard using Llama 3.3 70B.
        Returns a dict with keys: risk_gauge, advanced_analytics, asset_correlation.
        """
        if not self.llm:
            logger.info("Using deterministic fallback for dashboard explanations.")
            return self._fallback_dashboard_explanations(risk_class, features, portfolio_return, benchmark_return)

        try:
            vol = features.get('Annualized_Volatility', 0)
            var = features.get('Historical_VaR_95', 0)
            max_dd = features.get('Maximum_Drawdown', 0)
            sharpe = features.get('Sharpe', 0)
            perf_diff = portfolio_return - benchmark_return

            response = self.llm.invoke([
                SystemMessage(content="You are a financial risk analyst providing brief dashboard insights. Each insight must be 1-2 sentences, plain English, no jargon, strictly backward-looking. Never give investment advice."),
                HumanMessage(content=f"""Provide 3 brief dashboard insights for a {risk_class} risk portfolio:

Data: Volatility={vol:.2%}, VaR95={var:.2%}, MaxDD={max_dd:.2%}, Sharpe={sharpe:.2f}
Portfolio 6-month return: {portfolio_return:.2%}, vs Benchmark: {perf_diff:+.2%}

Format EXACTLY as (each under 30 words):
RISK_GAUGE: [insight about overall risk level]
ANALYTICS: [insight about portfolio vs benchmark]
CORRELATION: [insight about asset diversification]""")
            ])

            text = response.content.strip()
            result = {}
            for line in text.split('\n'):
                line = line.strip()
                if line.startswith('RISK_GAUGE:'):
                    result['risk_gauge'] = line.replace('RISK_GAUGE:', '').strip()
                elif line.startswith('ANALYTICS:'):
                    result['advanced_analytics'] = line.replace('ANALYTICS:', '').strip()
                elif line.startswith('CORRELATION:'):
                    result['asset_correlation'] = line.replace('CORRELATION:', '').strip()

            if len(result) == 3:
                logger.info("Dashboard explanations generated via LLM.")
                return result
            else:
                logger.warning("LLM response parsing incomplete. Using fallback.")
                return self._fallback_dashboard_explanations(risk_class, features, portfolio_return, benchmark_return)

        except Exception as e:
            logger.error(f"Dashboard explanations LLM call failed: {e}. Using fallback.")
            return self._fallback_dashboard_explanations(risk_class, features, portfolio_return, benchmark_return)


# Backward-compatible alias so existing imports still work
MockLLMAgent = LLMAgent
