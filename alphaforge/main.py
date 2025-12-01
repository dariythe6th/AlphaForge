# main.py
import asyncio
from src.agents.orchestrator import AlphaForgeCrew
from src.memory.long_term import LongTermMemory
from src.config import settings

class AlphaForgeSystem:
    def __init__(self):
        self.memory = LongTermMemory()
        self.crew = AlphaForgeCrew()
        self.is_evolving = False
        
    async def analyze_stocks(self, tickers: list, analysis_type: str = "comprehensive"):
        """Main entry point for stock analysis"""
        try:
            # Check memory for similar analyses
            context = await self.memory.get_context(tickers, analysis_type)
            
            # Execute the crew
            result = await self.crew.kickoff({
                "tickers": tickers,
                "analysis_type": analysis_type,
                "context": context
            })
            
            # Store results in memory
            await self.memory.store_analysis(result)
            
            # Trigger evolution if conditions met
            if not self.is_evolving:
                asyncio.create_task(self._trigger_evolution())
                
            return result
            
        except Exception as e:
            print(f"Analysis failed: {e}")
            # Fallback to rule-based analysis
            return await self._fallback_analysis(tickers)
    
    async def _trigger_evolution(self):
        """Trigger self-evolution process"""
        self.is_evolving = True
        try:
            await self.crew.strategy_optimizer.evolve_strategies()
        finally:
            self.is_evolving = False

async def main():
    system = AlphaForgeSystem()
    
    # Analyze Magnificent 7
    result = await system.analyze_stocks([
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA"
    ])
    
    print(f"Analysis completed: {result}")

if __name__ == "__main__":
    asyncio.run(main())