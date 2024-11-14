from langchain_experimental.utilities import PythonREPL
from typing import Optional, Dict, Any
import ast
import traceback
import re
import os
from datetime import datetime
import matplotlib.pyplot as plt
import json
import yfinance as yf
from langchain_groq import ChatGroq
from langchain_core.tools import Tool, tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
from dateutil import parser
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta, MO, TU, WE, TH, FR, SA, SU

matplotlib.use('Agg') 
class VisualizationREPL:
    def __init__(self, output_dir="./visualizations"):
        self.python_repl = PythonREPL()
        self.output_dir = output_dir
        self.last_plot_path = None

        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def generate_plot_filename(self) -> str:
        """Generate a unique filename for the plot."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.output_dir, f"plot_{timestamp}.png")

    def has_plotting_code(self, code: str) -> bool:
        """Check if code contains plotting-related commands."""
        plotting_keywords = [ 'matplotlib'
        ]
        return any(keyword in code for keyword in plotting_keywords)

    def save_current_plot(self) -> Optional[str]:
        """Save the current matplotlib plot if it exists."""
        if plt.get_fignums():  # Check if there are any figures
            filename = self.generate_plot_filename()
            plt.savefig(filename, bbox_inches='tight', dpi=300)
            plt.close('all')  # Close all figures to free memory
            self.last_plot_path = filename
            return filename
        return None

    def extract_code_blocks(self, text: str) -> list[str]:
        """Extract code from markdown-style code blocks or plain text."""
        # Try to find markdown code blocks
        code_blocks = re.findall(r'```python\n(.*?)\n```', text, re.DOTALL)
        if code_blocks:
            return code_blocks

        # Try to find any code blocks
        code_blocks = re.findall(r'```(.*?)```', text, re.DOTALL)
        if code_blocks:
            return code_blocks

        # If no code blocks found, treat entire text as code
        return [text.strip()]

    def clean_code(self, code: str) -> str:
        """Clean and standardize code string."""
        code = code.strip()
        code = re.sub(r'^```python\n', '', code)
        code = re.sub(r'\n```$', '', code)
        code = re.sub(r'^```\n', '', code)
        code = re.sub(r'\n```$', '', code)
        return code

    def validate_syntax(self, code: str) -> tuple[bool, Optional[str]]:
        """Validate Python code syntax before execution."""
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, f"Syntax Error: {str(e)}"
        except Exception as e:
            return False, f"Parsing Error: {str(e)}"

    def execute_visualization_code(self, code: str) -> Dict[str, Any]:
        """Execute code and handle visualization saving."""
        try:
            # Clear any existing plots
            plt.close('all')

            # Execute the code
            result = self.python_repl.run(code)

            # Check if there's a plot to save
            plot_path = None
            if self.has_plotting_code(code):
                plot_path = self.save_current_plot()

            return {
                "status": "success",
                "result": result,
                "plot_path": plot_path,
                "code": code
            }
        except Exception as e:
            return {
                "status": "error",
                "error_type": "runtime",
                "message": str(e),
                "traceback": traceback.format_exc(),
                "code": code
            }

    def run_with_auto_execute(self, input_text: str) -> str:
        """Main method to handle code execution and visualization."""
        # Extract code blocks
        code_blocks = self.extract_code_blocks(input_text)

        if not code_blocks:
            return "No executable code found in the input."

        # Process each code block
        results = []
        for code in code_blocks:
            # Clean the code
            clean_code = self.clean_code(code)

            # Validate syntax
            is_valid, syntax_error = self.validate_syntax(clean_code)
            if not is_valid:
                results.append(f"Syntax Error in code:\n{code}\nError: {syntax_error}")
                continue

            # Execute the code
            result = self.execute_visualization_code(clean_code)

            if result["status"] == "success":
                output = f"Code executed successfully."
                if result["result"]:
                    output += f"\nOutput:\n{result['result']}"
                if result.get("plot_path"):
                    output += f"\nPlot saved to: {result['plot_path']}"
                results.append(output)
            else:
                results.append(f"Runtime Error:\n{result['message']}")

        return "\n\n".join(results)

def create_visualization_tool(output_dir="./visualizations") -> Tool:
    """Create a Python REPL tool with visualization support."""
    viz_repl = VisualizationREPL(output_dir=output_dir)

    return Tool(
        name="python_visualization_repl",
        description="""A Python REPL that automatically executes code and saves matplotlib plots for visualization as PNG files.
        Especially useful for data visualization tasks. The tool will execute the code,
        display any output, and save any generated plots as PNG files.""",
        func=viz_repl.run_with_auto_execute
    )

visualization_tool = create_visualization_tool()


class DateParserTool:
    def weekday_to_relativedelta(self, weekday):
        # Map weekday names to relativedelta objects
        weekdays_map = {
            "monday": MO,
            "tuesday": TU,
            "wednesday": WE,
            "thursday": TH,
            "friday": FR,
            "saturday": SA,
            "sunday": SU,
        }
        return weekdays_map.get(weekday)

    def run(self, query: str) -> str:
        """
        Convert date-related queries to date string.

        Args:
            query (str): Date-related query (e.g., '2 days', 'last monday', 'yesterday')

        Returns:
            str: Date in format 'YYYY-MM-DD'
        """
        today = datetime.now()
        query = query.lower()

        match_numeric = re.search(r"(\d+)\s*(day|days|week|weeks|month|months|year|years)", query)
        if match_numeric:
            num = int(match_numeric.group(1))
            unit = match_numeric.group(2)

            if "day" in unit:
                return (today - timedelta(days=num)).strftime("%Y-%m-%d")
            elif "week" in unit:
                return (today - timedelta(weeks=num)).strftime("%Y-%m-%d")
            elif "month" in unit:
                return (today - relativedelta(months=num)).strftime("%Y-%m-%d")
            elif "year" in unit:
                return (today - relativedelta(years=num)).strftime("%Y-%m-%d")

        match = re.search(r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)", query)
        weekday = match.group(0) if match else None

        if query == "today":
            return today.strftime("%Y-%m-%d")
        elif query == "yesterday":
            return (today - timedelta(days=1)).strftime("%Y-%m-%d")
        elif weekday:
            rd_weekday = self.weekday_to_relativedelta(weekday)
            if rd_weekday:
                if query == weekday:
                    return (today - relativedelta(weekday=rd_weekday(-1))).strftime("%Y-%m-%d")
                elif "last" in query:
                    if today.weekday() != rd_weekday.weekday:
                        return (today - relativedelta(weekday=rd_weekday(-1))).strftime("%Y-%m-%d")
                    else:
                        return (today - relativedelta(weeks=1, weekday=rd_weekday(-1))).strftime("%Y-%m-%d")

        return "Invalid date query"

        # Try parsing as an absolute date
        try:
            parsed_date = parse(query)
            return (today - parsed_date).days
        except (ValueError, TypeError):
            raise ValueError(f"Unable to parse date from query: {query}")




class FinancialAnalysisAgent:
    def __init__(self, api_key):
        """Initialize the Financial Analysis Agent with API key and tools."""
        self.api_key = api_key
        self.llm = ChatGroq(
            model_name="llama3-70b-8192",
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
                    "52_week_high": company_info_full.get("fiftyTwoWeekHigh"),
                    "52_week_low": company_info_full.get("fiftyTwoWeekLow"),}
                return company_info_cleaned, f"Company Information: {symbol}"
            except Exception as e:
                return f"Error fetching company profile for {symbol}: {e}"


        def historical_stock_price_df(symbol: str,user:int) -> str:
            try:
                stock = yf.Ticker(symbol)
                historical_data = stock.history(period="5y")

                if historical_data.empty:
                    return f"No historical data found for {symbol}."

                historical_data.reset_index(inplace=True)
                historical_data.rename(columns={'Date':'date','Close':'close'},inplace=True)
                historical_data.drop(columns=['Dividends','Stock Splits'],inplace=True)
                stock_price = historical_data.loc[:user]
                return stock_price
            except Exception as e:
                return f"Error fetching stock price for {symbol} : {e}"

        @tool
        def get_historical_stock_price(symbol: str,query: str) -> str:
            """Get the historical stock price of a stock

            Args:

            symbol (str): The stock ticker symbol
            query (str): time priod example: 1 week,2 month, last 2 days,last friday

            """
            user = DateParserTool().run(query)
            return historical_stock_price_df(symbol,user)



        @tool
        def risk_assessment(ticker: str, benchmark: str = "^NSEI", period: str = "5y"):
            """
           Perform risk assessment for a given stock.

           Args:
           ticker (str): The stock ticker symbol.
           benchmark (str): Benchmark index for comparison (default: S&P 500).
           period (str): Time period for analysis.

           Returns:
           dict: Risk assessment results.
           """
            stock = yf.Ticker(ticker)
            benchmark_index = yf.Ticker(benchmark)

            stock_data = stock.history(period=period)['Close']
            benchmark_data = benchmark_index.history(period=period)['Close']
            common_dates = stock_data.index.intersection(benchmark_data.index)
            stock_data = stock_data[common_dates]
            benchmark_data = benchmark_data[common_dates]

      
            stock_returns = stock_data.pct_change().dropna()
            benchmark_returns = benchmark_data.pct_change().dropna()

         
            covariance = np.cov(stock_returns, benchmark_returns)[0][1]
            benchmark_variance = np.var(benchmark_returns)
            beta = covariance / benchmark_variance

    
            risk_free_rate = 0.06
            excess_returns = stock_returns - risk_free_rate
            sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()

            var_95 = np.percentile(stock_returns, 5)

            # Calculate Maximum Drawdown
            cumulative_returns = (1 + stock_returns).cumprod()
            max_drawdown = (cumulative_returns.cummax() - cumulative_returns).max()

            return {
            "ticker": ticker,
            "beta": beta,
            "sharpe_ratio": sharpe_ratio,
            "value_at_risk_95": var_95,
            "max_drawdown": max_drawdown,
            "volatility": stock_returns.std() * np.sqrt(252)
             }



        @tool
        def get_key_comparision_ratios(symbol: str) -> str:
            """Use this function to get BookValue , PEG Ratio , Beta ,Debt/Equity , Trailing_EPS, Trailing_P/E Ratio of a stock.

            Args:
            symbol (str): The stock symbol.

            Returns:
            str: JSON containing  BookValue , PEG Ratio , Beta ,Debt/Equity Trailing_EPS, Trailing_P/E Ratio.

            """
            try:
                stock = yf.Ticker(symbol)
                info = stock.info
                fundamentals = {
                    "Forward_P/E_ratio": info.get("forwardPE", "N/A"),
                    "Trailing_P/E_ratio": info.get("trailingPE", "N/A"),
                    "Forward_P/B_ratio": info.get("forwardPB", "N/A"),
                    "P/B Ratio": info.get("priceToBook", "N/A"),
                    "Dividend Yield": info.get("dividendYield", "N/A"),
                    "Trailing_EPS": info.get("trailingEps", "N/A"),
                    "forward_EPS": info.get("forwardEps", "N/A"),
                    "Beta": info.get("beta", "N/A"),
                    "BookValue": info.get("bookValue", "N/A"),
                    "Debt_Equity_ratio": info.get("debtToEquity", "N/A"),
                    "PEG_Ratio" : info.get("pegRatio", "N/A")
                }
                return json.dumps(fundamentals, indent=2)
            except Exception as e:
                return f"Error getting fundamentals for {symbol}: {e}"

        @tool
        def get_quaterly_income_statement(symbol: str) -> str:
          """
          Return the income statement of the stock.

          Args:
          ticker (str): the stock ticker to be given to yfinance

          """
          try:
            stock = yf.Ticker(symbol)
            income_statement = pd.DataFrame(stock.quarterly_income_stmt)
            return "Quaterly Income Statement: \n" + income_statement.to_string()
          except Exception as e:
            return f"Error getting Quaterly Income  Statement for {symbol}: {e}"

        @tool
        def get_key_financial_ratios(symbol: str) -> str:
          """Use this function to get key financial ratios for a given stock symbol.

          Args:
          symbol (str): The stock symbol.

          Returns:
          dict: JSON containing key financial ratios.
          """
          try:
            stock = yf.Ticker(symbol)
            key_ratios = stock.info
            return json.dumps(key_ratios, indent=2)
          except Exception as e:
            return f"Error fetching key financial ratios for {symbol}: {e}"

        @tool
        def get_analyst_recommendations(symbol: str) -> str:
          """Use this function to get analyst recommendations for a given stock symbol.

          Args:
          symbol (str): The stock symbol.

          Returns:
          str: JSON containing analyst recommendations.
          """
          try:
            stock = yf.Ticker(symbol)
            recommendations = stock.recommendations
            return recommendations.to_json(orient="index")
          except Exception as e:
            return f"Error fetching analyst recommendations for {symbol}: {e}"

        self.tools = [
            get_company_info,
            get_key_comparision_ratios,
            get_quaterly_income_statement,
            get_analyst_recommendations,
            visualization_tool,
            risk_assessment]

    def setup_agent(self):
        """Configure the agent with prompt template and tools."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a sophisticated financial analysis agent capable of:
            Analyzing complex financial data and also visualising the data.
            Use get_key_comparision_ratios tool to fetch any find of ratio for a stock
            USE MATPLOTLIB library  for any visualisation"""),
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


# Example usage
if __name__ == "__main__":
    api_key = "gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    agent = FinancialAnalysisAgent(api_key)
    result = agent.analyze("""Fetch Trailing EPS Ratio,BookValue Ratio and Trailing P/E Ratio of BEL.NS and plot a bar graph""")


    print(result)

