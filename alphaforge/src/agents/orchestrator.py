# src/agents/orchestrator.py
from crewai import Agent, Crew, Process, Task
from langchain.tools import Tool
from langchain_community.chat_models import ChatOpenAI
from src.mcp_client import MCPClient
from src.memory.short_term import ShortTermMemory
import asyncio

class AlphaForgeCrew:
    def __init__(self):
        self.llm_ultra = ChatOpenAI(model="gpt-4", temperature=0.1)
        self.llm_fast = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
        self.mcp_client = MCPClient()
        self.short_memory = ShortTermMemory()
        
        # Initialize agents
        self.orchestrator = self._create_orchestrator()
        self.data_agent = self._create_data_acquisition_agent()
        self.analyst_agent = self._create_core_analyst_agent()
        self.report_agent = self._create_report_generator_agent()
        self.strategy_optimizer = self._create_strategy_optimizer()
        
    def _create_orchestrator(self):
        return Agent(
            role="Senior Portfolio Manager",
            goal="Orchestrate the entire financial analysis workflow and ensure high-quality investment recommendations",
            backstory="""You are an experienced portfolio manager at a top hedge fund with 20 years of experience 
            in quantitative analysis and risk management. You coordinate teams of analysts to produce 
            institutional-grade investment research.""",
            tools=[self.mcp_client.get_tool()],
            llm=self.llm_ultra,
            memory=True,
            verbose=True
        )
    
    def _create_data_acquisition_agent(self):
        return Agent(
            role="Data Acquisition Specialist",
            goal="Fetch and preprocess multi-modal financial data from various sources",
            backstory="""You are a data engineer specialized in financial data pipelines. You handle real-time 
            data feeds, API integrations, and data quality assurance.""",
            tools=[self.mcp_client.get_tool("data_acquisition")],
            llm=self.llm_fast,
            memory=True,
            verbose=True
        )
    
    def _create_core_analyst_agent(self):
        return Agent(
            role="Senior Financial Analyst",
            goal="Perform comprehensive technical, fundamental, and sentiment analysis on financial data",
            backstory="""You are a CFA charterholder with expertise in both quantitative and qualitative analysis. 
            You combine algorithmic outputs with strategic thinking to generate actionable insights.""",
            tools=[
                self.mcp_client.get_tool("algorithm_server"),
                Tool(
                    name="hybrid_rag_retrieval",
                    func=self._retrieve_financial_context,
                    description="Retrieve relevant financial context from knowledge base"
                )
            ],
            llm=self.llm_ultra,
            memory=True,
            verbose=True
        )
    
    def _create_report_generator_agent(self):
        return Agent(
            role="Investment Report Writer",
            goal="Create comprehensive, visually appealing investment reports with clear recommendations",
            backstory="""You are a financial writer who translates complex analysis into clear, actionable 
            reports for portfolio managers and clients.""",
            tools=[self.mcp_client.get_tool("visualization")],
            llm=self.llm_ultra,
            memory=True,
            verbose=True
        )
    
    def _create_strategy_optimizer(self):
        return Agent(
            role="Strategy Optimization Specialist",
            goal="Continuously improve analysis strategies based on performance feedback",
            backstory="""You are a machine learning researcher focused on optimizing financial analysis 
            strategies through reinforcement learning and pattern recognition.""",
            tools=[self.mcp_client.get_tool()],
            llm=self.llm_ultra,
            memory=True,
            verbose=True
        )
    
    def _retrieve_financial_context(self, query: str):
        """Hybrid RAG retrieval function"""
        # Implementation details in RAG section
        pass
    
    async def kickoff(self, inputs: dict):
        """Execute the complete analysis workflow"""
        
        # Define tasks with dependencies
        data_task = Task(
            description=f"""
            Acquire and preprocess multi-modal data for tickers: {inputs['tickers']}
            Include: price data, news sentiment, SEC filings, and earnings call materials.
            Output cleaned, normalized data ready for analysis.
            """,
            agent=self.data_agent,
            expected_output="Structured dataset with price history, sentiment scores, and fundamental data",
            async_execution=True
        )
        
        analysis_task = Task(
            description=f"""
            Perform comprehensive analysis on the prepared data for {inputs['tickers']}:
            1. Technical Analysis (RSI, MACD, Bollinger Bands, Volume trends)
            2. Fundamental Analysis (P/E, P/B, ROE, Growth rates, Cash flow)
            3. Sentiment Analysis (News tone, Market sentiment)
            4. Risk Assessment (VaR, Volatility, Drawdown analysis)
            Incorporate context from financial knowledge base.
            """,
            agent=self.analyst_agent,
            expected_output="Detailed analysis with buy/hold/sell recommendations and target prices",
            context=[data_task]
        )
        
        report_task = Task(
            description=f"""
            Generate a comprehensive investment report for {inputs['tickers']} including:
            - Executive summary with key recommendations
            - Detailed analysis sections with supporting data
            - Risk-adjusted return projections
            - Visualizations (price charts, technical indicators, sentiment trends)
            - Portfolio allocation suggestions
            """,
            agent=self.report_agent,
            expected_output="Professional PDF report with charts and tables",
            context=[analysis_task],
            output_file="reports/investment_analysis.pdf"
        )
        
        # Create and run crew
        crew = Crew(
            agents=[self.orchestrator, self.data_agent, self.analyst_agent, self.report_agent],
            tasks=[data_task, analysis_task, report_task],
            process=Process.hierarchical,
            manager_agent=self.orchestrator,
            memory=True,
            verbose=True
        )
        
        return await crew.kickoff(inputs)