import os
import json
import yfinance as yf
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage

# Set page config
st.set_page_config(
    page_title="Financial Analysis Agent",
    page_icon="💹",
    layout="wide"
)

# Configure Groq API
class FinancialAnalysisAgent:
    def __init__(self, api_key):
        """Initialize the Financial Analysis Agent with API key and tools."""
        self.api_key = api_key
        os.environ["GROQ_API_KEY"] = api_key
        self.llm = ChatGroq(
            model_name="llama-3.1-70b-versatile",
            api_key=self.api_key
        )
        self.setup_tools()
        self.setup_agent()

    def setup_tools(self):
        """Set up financial analysis tools."""

        @tool
        def get_company_info(symbol: str) -> str:
            """Get company information and overview for a given stock symbol."""
            try:
                company_info_full = yf.Ticker(symbol).info
                if company_info_full is None:
                    return f"Could not fetch company info for {symbol}"

                company_info_cleaned = {
                    "Name": company_info_full.get("shortName"),
                    "Symbol": company_info_full.get("symbol"),
                    "Current Stock Price": f"{company_info_full.get('regularMarketPrice', company_info_full.get('currentPrice'))} {company_info_full.get('currency', 'USD')}",
                    "Market Cap": f"{company_info_full.get('marketCap', company_info_full.get('enterpriseValue'))} {company_info_full.get('currency', 'USD')}",
                    "Sector": company_info_full.get("sector"),
                    "Industry": company_info_full.get("industry"),
                    "Website": company_info_full.get("website"),
                    "Summary": company_info_full.get("longBusinessSummary"),
                    "Analyst Recommendation": company_info_full.get("recommendationKey"),
                    "Number Of Analyst Opinions": company_info_full.get("numberOfAnalystOpinions"),
                    "Key Metrics": {
                        "EPS": company_info_full.get("trailingEps"),
                        "P/E Ratio": company_info_full.get("trailingPE"),
                        "52 Week Range": f"{company_info_full.get('fiftyTwoWeekLow')} - {company_info_full.get('fiftyTwoWeekHigh')}",
                        "50 Day Average": company_info_full.get("fiftyDayAverage"),
                        "200 Day Average": company_info_full.get("twoHundredDayAverage"),
                    },
                    "Financial Health": {
                        "Total Cash": company_info_full.get("totalCash"),
                        "Free Cash Flow": company_info_full.get("freeCashflow"),
                        "Operating Cash Flow": company_info_full.get("operatingCashflow"),
                        "EBITDA": company_info_full.get("ebitda"),
                        "Revenue Growth": company_info_full.get("revenueGrowth"),
                        "Gross Margins": company_info_full.get("grossMargins"),
                        "EBITDA Margins": company_info_full.get("ebitdaMargins"),
                    }
                }
                return json.dumps(company_info_cleaned, indent=2)
            except Exception as e:
                return f"Error fetching company profile for {symbol}: {e}"

        @tool
        def get_historical_stock_prices(symbol: str, period: str = "1mo", interval: str = "1d") -> str:
            """Get historical stock prices with customizable period and interval."""
            try:
                stock = yf.Ticker(symbol)
                historical_price = stock.history(period=period, interval=interval)
                return historical_price.to_json(orient="index")
            except Exception as e:
                return f"Error fetching historical prices for {symbol}: {e}"

        @tool
        def get_stock_fundamentals(symbol: str) -> str:
            """Get key fundamental data for stock analysis."""
            try:
                stock = yf.Ticker(symbol)
                info = stock.info
                fundamentals = {
                    "symbol": symbol,
                    "company_name": info.get("longName", ""),
                    "sector": info.get("sector", ""),
                    "industry": info.get("industry", ""),
                    "market_cap": info.get("marketCap", "N/A"),
                    "pe_ratio": info.get("forwardPE", "N/A"),
                    "pb_ratio": info.get("priceToBook", "N/A"),
                    "dividend_yield": info.get("dividendYield", "N/A"),
                    "eps": info.get("trailingEps", "N/A"),
                    "beta": info.get("beta", "N/A"),
                    "52_week_high": info.get("fiftyTwoWeekHigh", "N/A"),
                    "52_week_low": info.get("fiftyTwoWeekLow", "N/A"),
                }
                return json.dumps(fundamentals, indent=2)
            except Exception as e:
                return f"Error getting fundamentals for {symbol}: {e}"
        
        @tool
        def get_income_statements(symbol: str) -> str:
            """Use this function to get income statements for a given stock symbol."""
            try:
                stock = yf.Ticker(symbol)
                financials = stock.financials
                return financials.to_json(orient="index")
            except Exception as e:
                return f"Error fetching income statements for {symbol}: {e}"

        @tool
        def get_key_financial_ratios(symbol: str) -> str:
            """Use this function to get key financial ratios for a given stock symbol."""
            try:
                stock = yf.Ticker(symbol)
                key_ratios = stock.info
                return json.dumps(key_ratios, indent=2)
            except Exception as e:
                return f"Error fetching key financial ratios for {symbol}: {e}"

        @tool
        def get_analyst_recommendations(symbol: str) -> str:
            """Use this function to get analyst recommendations for a given stock symbol."""
            try:
                stock = yf.Ticker(symbol)
                recommendations = stock.recommendations
                return recommendations.to_json(orient="index")
            except Exception as e:
                return f"Error fetching analyst recommendations for {symbol}: {e}"

        self.tools = [
            get_company_info,
            get_historical_stock_prices,
            get_stock_fundamentals,
            get_income_statements,
            get_key_financial_ratios,
            get_analyst_recommendations
        ]

    def setup_agent(self):
        """Configure the agent with prompt template and tools."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a sophisticated financial analysis agent capable of:
            Analyzing complex financial data."""),
            MessagesPlaceholder(variable_name="input"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_tool_calling_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )

        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True
        )

    def analyze(self, query: str):
        """Execute financial analysis based on user query."""
        return self.agent_executor.invoke({
            "input": [HumanMessage(content=query)]
        })

# Streamlit UI
def main():
    st.title("🤖 Financial Analysis Agent")
    st.write("Enter your API key and ask questions about stocks!")

    # Sidebar for API key
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("Enter your Groq API Key:", type="password")
        st.markdown("---")
        st.markdown("""
        ### Sample Queries:
        - Analyze the fundamental metrics for AAPL
        - Compare the financial health of MSFT and GOOGL
        - Get historical price analysis for TSLA for the last 3 months
        - Prepare a comprehensive report on NVDA's financial performance
        """)

    # Main content
    if not api_key:
        st.warning("Please enter your Groq API key in the sidebar to continue.")
        return

    # Initialize agent
    try:
        agent = FinancialAnalysisAgent(api_key)
    except Exception as e:
        st.error(f"Error initializing agent: {str(e)}")
        return

    # Query input
    query = st.text_area("Enter your analysis query:", height=100)
    
    if st.button("Analyze"):
        if not query:
            st.warning("Please enter a query.")
            return
        
        with st.spinner("Analyzing..."):
            try:
                result = agent.analyze(query)
                
                # Display results in an expandable section
                with st.expander("Analysis Results", expanded=True):
                    st.markdown(result['output'])
                    
                    # Try to parse any JSON in the output for better formatting
                    try:
                        # Look for JSON strings in the output and format them
                        import re
                        json_strings = re.findall(r'{.*}', result['output'], re.DOTALL)
                        for json_str in json_strings:
                            try:
                                parsed_json = json.loads(json_str)
                                st.json(parsed_json)
                            except:
                                continue
                    except:
                        pass

            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    main()
