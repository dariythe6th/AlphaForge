# src/agents/orchestrator.py
import asyncio
import logging
from typing import Dict, Any, List
from datetime import datetime

from crewai import Agent, Crew, Process, Task
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain.callbacks import FileCallbackHandler

from src.config import settings
from src.mcp_client import MCPClient
from src.memory.short_term import ShortTermMemory
from src.rag.hybrid_retriever import HybridRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlphaForgeCrew:
    def __init__(self):
        # Initialize LLMs with proper configuration
        self.llm_ultra = ChatOpenAI(
            model_name="gpt-4",
            temperature=0.1,
            max_tokens=4000,
            api_key=settings.OPENAI_API_KEY
        )
        self.llm_fast = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.1,
            max_tokens=2000,
            api_key=settings.OPENAI_API_KEY
        )

        # Initialize clients
        self.mcp_client = MCPClient()
        self.short_memory = ShortTermMemory()
        self.retriever = HybridRetriever()

        # Initialize agents
        self.orchestrator = self._create_orchestrator()
        self.data_agent = self._create_data_acquisition_agent()
        self.analyst_agent = self._create_core_analyst_agent()
        self.report_agent = self._create_report_generator_agent()
        self.strategy_optimizer = None  # Will be initialized when needed

        # Set up logging
        self.handler = FileCallbackHandler("logs/crewai.log")

        logger.info("AlphaForge crew initialized")

    def _create_orchestrator(self) -> Agent:
        """Create the orchestrator agent"""
        return Agent(
            role="Senior Portfolio Manager",
            goal="Orchestrate the entire financial analysis workflow and ensure high-quality investment recommendations",
            backstory="""You are an experienced portfolio manager at a top hedge fund with 20 years of experience 
            in quantitative analysis and risk management. You coordinate teams of analysts to produce 
            institutional-grade investment research.""",
            tools=[self._create_mcp_tool()],
            llm=self.llm_ultra,
            memory=True,
            verbose=True,
            allow_delegation=True,
            max_iter=3
        )

    def _create_data_acquisition_agent(self) -> Agent:
        """Create the data acquisition agent"""
        return Agent(
            role="Data Acquisition Specialist",
            goal="Fetch and preprocess multi-modal financial data from various sources",
            backstory="""You are a data engineer specialized in financial data pipelines. You handle real-time 
            data feeds, API integrations, and data quality assurance.""",
            tools=[self._create_mcp_tool("data_acquisition")],
            llm=self.llm_fast,
            memory=True,
            verbose=True,
            allow_delegation=False,
            max_iter=2
        )

    def _create_core_analyst_agent(self) -> Agent:
        """Create the core analyst agent"""
        return Agent(
            role="Senior Financial Analyst",
            goal="Perform comprehensive technical, fundamental, and sentiment analysis on financial data",
            backstory="""You are a CFA charterholder with expertise in both quantitative and qualitative analysis. 
            You combine algorithmic outputs with strategic thinking to generate actionable insights.""",
            tools=[
                self._create_mcp_tool("algorithm_server"),
                Tool(
                    name="hybrid_rag_retrieval",
                    func=self._retrieve_financial_context_sync,
                    description="Retrieve relevant financial context from knowledge base"
                )
            ],
            llm=self.llm_ultra,
            memory=True,
            verbose=True,
            allow_delegation=False,
            max_iter=5
        )

    def _create_report_generator_agent(self) -> Agent:
        """Create the report generator agent"""
        return Agent(
            role="Investment Report Writer",
            goal="Create comprehensive, visually appealing investment reports with clear recommendations",
            backstory="""You are a financial writer who translates complex analysis into clear, actionable 
            reports for portfolio managers and clients.""",
            tools=[self._create_mcp_tool("visualization")],
            llm=self.llm_ultra,
            memory=True,
            verbose=True,
            allow_delegation=False,
            max_iter=3
        )

    def _create_mcp_tool(self, server_type: str = "general") -> Tool:
        """Create MCP client tool"""
        return Tool(
            name=f"mcp_{server_type}_client",
            func=self.mcp_client.call_tool,
            description=f"Call {server_type} MCP server for data or processing"
        )

    def _retrieve_financial_context_sync(self, query: str) -> str:
        """Synchronous wrapper for RAG retrieval"""
        try:
            # Run async function in event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(
                self.retriever.retrieve(query, top_k=3)
            )
            loop.close()

            if results:
                context = "\n\n".join([
                    f"Source: {r.get('type', 'unknown')}\n"
                    f"Confidence: {r.get('score', 0):.2f}\n"
                    f"Content: {r.get('content', '')}"
                    for r in results[:3]
                ])
                return f"Retrieved context:\n{context}"
            else:
                return "No relevant context found in knowledge base."

        except Exception as e:
            logger.error(f"Error retrieving financial context: {e}")
            return f"Error retrieving context: {e}"

    async def kickoff(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the complete analysis workflow asynchronously"""
        try:
            logger.info(f"Starting analysis workflow for {inputs.get('tickers', [])}")

            # Validate inputs
            tickers = inputs.get('tickers', [])
            if not tickers:
                raise ValueError("No tickers provided")

            # Limit number of tickers for performance
            tickers = tickers[:10]  # Max 10 tickers

            analysis_type = inputs.get('analysis_type', 'comprehensive')
            context = inputs.get('context', [])

            # Create tasks with proper dependencies
            data_task = Task(
                description=f"""
                Acquire and preprocess multi-modal data for tickers: {', '.join(tickers)}

                Required data:
                1. Price data for the last year (daily)
                2. News articles and sentiment for the last 30 days
                3. Latest SEC filings (10-K or 10-Q)
                4. Any available earnings call materials

                Ensure data is cleaned, normalized, and ready for analysis.
                Handle any missing data gracefully and log issues.
                """,
                agent=self.data_agent,
                expected_output="""Structured dataset containing:
                - Clean price history with OHLCV data
                - Sentiment scores and key news articles
                - Fundamental metrics from SEC filings
                - Timestamps and data quality metrics""",
                async_execution=True,
                output_json=True
            )

            analysis_task = Task(
                description=f"""
                Perform comprehensive analysis on the prepared data for {', '.join(tickers)}:

                Analysis Components:
                1. Technical Analysis:
                   - Calculate RSI, MACD, Bollinger Bands
                   - Identify trends and support/resistance levels
                   - Volume analysis and indicators

                2. Fundamental Analysis:
                   - Calculate key ratios (P/E, P/B, ROE, etc.)
                   - Growth rate analysis
                   - Cash flow and balance sheet health

                3. Sentiment Analysis:
                   - Aggregate news sentiment scores
                   - Social media and market sentiment indicators

                4. Risk Assessment:
                   - Calculate volatility and Value at Risk (VaR)
                   - Maximum drawdown analysis
                   - Stress testing scenarios

                5. Context Integration:
                   - Incorporate relevant knowledge from financial database
                   - Consider macroeconomic factors
                   - Compare with industry peers

                Output clear, actionable insights with confidence scores.
                """,
                agent=self.analyst_agent,
                expected_output="""Comprehensive analysis containing:
                - Technical indicators and signals
                - Fundamental ratios and valuations
                - Sentiment scores and trends
                - Risk metrics and assessments
                - Investment recommendations with confidence scores
                - Target price ranges
                - Key risks and mitigations""",
                context=[data_task],
                output_json=True
            )

            report_task = Task(
                description=f"""
                Generate a comprehensive investment report for {', '.join(tickers)}:

                Report Structure:
                1. Executive Summary (1-2 pages)
                   - Key findings and recommendations
                   - Expected returns and risk levels
                   - Time horizon and investment thesis

                2. Detailed Analysis Sections:
                   - Technical Analysis with charts
                   - Fundamental Analysis with tables
                   - Sentiment Analysis with trends
                   - Risk Assessment with metrics

                3. Visualizations:
                   - Price charts with technical indicators
                   - Comparative analysis charts
                   - Risk-return scatter plots
                   - Sentiment timeline charts

                4. Recommendations:
                   - Specific buy/hold/sell recommendations
                   - Portfolio allocation suggestions
                   - Entry and exit strategies
                   - Risk management guidelines

                5. Appendices:
                   - Data sources and methodology
                   - Key assumptions
                   - Glossary of terms

                Ensure the report is professional, data-driven, and actionable.
                Use clear, concise language suitable for institutional investors.
                """,
                agent=self.report_agent,
                expected_output="Professional investment report in PDF format with embedded charts and tables",
                context=[analysis_task],
                output_file=f"reports/analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            )

            # Create and configure crew
            crew = Crew(
                agents=[self.orchestrator, self.data_agent, self.analyst_agent, self.report_agent],
                tasks=[data_task, analysis_task, report_task],
                process=Process.hierarchical,
                manager_agent=self.orchestrator,
                memory=True,
                verbose=True,
                callbacks=[self.handler]
            )

            # Execute workflow
            logger.info("Starting crew execution")
            result = await asyncio.to_thread(crew.kickoff, inputs)

            # Add metadata to result
            result['metadata'] = {
                'tickers_analyzed': tickers,
                'analysis_type': analysis_type,
                'timestamp': datetime.now().isoformat(),
                'crew_version': '1.0',
                'processing_time': result.get('processing_time', 0)
            }

            logger.info(f"Analysis completed successfully for {tickers}")
            return result

        except Exception as e:
            logger.error(f"Error in analysis workflow: {e}")
            # Fallback to simplified analysis
            return await self._fallback_analysis(inputs)

    async def _fallback_analysis(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback analysis when main workflow fails"""
        logger.warning("Using fallback analysis")

        tickers = inputs.get('tickers', [])

        return {
            'status': 'fallback',
            'tickers': tickers,
            'recommendation': 'HOLD',
            'confidence': 0.5,
            'analysis': 'Fallback analysis due to system error',
            'timestamp': datetime.now().isoformat(),
            'suggested_actions': ['Review system logs', 'Retry with fewer tickers']
        }