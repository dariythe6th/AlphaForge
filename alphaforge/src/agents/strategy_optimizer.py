# src/agents/strategy_optimizer.py
from crewai import Agent, Task
from langchain_openai import ChatOpenAI
import asyncio
from typing import Dict, Any, List
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StrategyOptimizerAgent:
    def __init__(self, memory, llm_ultra=None):
        self.memory = memory
        if llm_ultra is None:
            from src.config import settings
            self.llm_ultra = ChatOpenAI(
                model_name="gpt-4",
                temperature=0.1,
                api_key=settings.OPENAI_API_KEY
            )
        else:
            self.llm_ultra = llm_ultra

        self.agent = self._create_agent()
        logger.info("Strategy Optimizer Agent initialized")

    def _create_agent(self) -> Agent:
        """Create the strategy optimizer agent"""
        return Agent(
            role="Strategy Optimization Specialist",
            goal="Continuously improve analysis strategies based on performance feedback and market changes",
            backstory="""You are a machine learning researcher and quantitative analyst with expertise in 
            reinforcement learning and evolutionary algorithms. You analyze past performance to identify 
            successful patterns and optimize the system's analysis strategies.""",
            llm=self.llm_ultra,
            verbose=True,
            allow_delegation=False
        )

    async def evolve_strategies(self) -> List[Dict[str, Any]]:
        """Main evolution loop"""
        try:
            logger.info("Starting strategy evolution")

            # Analyze past performance
            patterns = await self.memory.get_evolution_patterns(min_score=0.8)

            if not patterns:
                logger.warning("No successful patterns found for evolution")
                return []

            # Generate improved strategies
            improved_strategies = await self._generate_improved_strategies(patterns)

            # Store new strategies
            for strategy in improved_strategies:
                await self._store_strategy(strategy)

            logger.info(f"Generated {len(improved_strategies)} improved strategies")
            return improved_strategies

        except Exception as e:
            logger.error(f"Strategy evolution failed: {e}")
            return []

    async def _generate_improved_strategies(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate new strategies based on successful patterns"""
        try:
            # Create analysis prompt for LLM
            analysis_prompt = f"""
            Analyze these successful financial analysis patterns and generate improved strategies:

            {json.dumps(patterns, indent=2)}

            Consider:
            1. What common elements make these patterns successful?
            2. How can we combine successful elements in new ways?
            3. What market conditions were present during successful analyses?
            4. How can we adapt strategies to current market volatility?
            5. What new data sources or techniques could improve accuracy?

            Return a list of improved strategy configurations in JSON format.
            Each strategy should include:
            - name: Descriptive name
            - approach: Description of the analysis approach
            - data_sources: List of data sources to prioritize
            - risk_adjustment: How to adjust risk calculations
            - conditions: Market conditions where this strategy works best
            - expected_improvement: Expected improvement percentage (0-100)
            """

            # For now, return mock strategies
            # In production, this would call the LLM
            return [
                {
                    "name": "Hybrid Momentum-Fundamental Strategy",
                    "approach": "Combine short-term momentum signals with long-term fundamental value",
                    "data_sources": ["price_momentum", "earnings_growth", "market_sentiment"],
                    "risk_adjustment": "Dynamic volatility scaling based on market regime",
                    "conditions": ["bull_markets", "high_volatility"],
                    "expected_improvement": 15.5
                },
                {
                    "name": "Sentiment-Adjusted Technical Analysis",
                    "approach": "Adjust technical indicators based on real-time news sentiment",
                    "data_sources": ["technical_indicators", "news_sentiment", "social_media"],
                    "risk_adjustment": "Increase position sizing when sentiment confirms technicals",
                    "conditions": ["news_heavy_periods", "earnings_season"],
                    "expected_improvement": 12.3
                }
            ]

        except Exception as e:
            logger.error(f"Error generating improved strategies: {e}")
            return []

    async def _store_strategy(self, strategy: Dict[str, Any]) -> bool:
        """Store a new strategy in memory"""
        try:
            # Store in memory system
            await self.memory.store_analysis({
                "type": "strategy",
                "name": strategy.get("name"),
                "content": strategy,
                "timestamp": __import__("datetime").datetime.now().isoformat()
            })
            return True
        except Exception as e:
            logger.error(f"Error storing strategy: {e}")
            return False