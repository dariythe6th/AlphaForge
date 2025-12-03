# main.py
import asyncio
import logging
import signal
import sys
from datetime import datetime
from typing import List

from src.agents.orchestrator import AlphaForgeCrew
from src.memory.long_term import LongTermMemory
from src.config import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/alphaforge.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AlphaForgeSystem:
    def __init__(self):
        logger.info("Initializing AlphaForge System")

        self.memory = LongTermMemory(
            redis_url=settings.REDIS_URL,
            neo4j_uri=settings.NEO4J_URL,
            neo4j_user=settings.NEO4J_USER,
            neo4j_password=settings.NEO4J_PASSWORD
        )

        self.crew = AlphaForgeCrew()
        self.is_evolving = False
        self.analysis_count = 0

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info("AlphaForge System initialized successfully")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.shutdown()
        sys.exit(0)

    async def analyze_stocks(self, tickers: List[str],
                             analysis_type: str = "comprehensive") -> dict:
        """Main entry point for stock analysis"""
        try:
            logger.info(f"Starting analysis for {tickers}, type: {analysis_type}")

            # Validate tickers
            if not tickers:
                raise ValueError("No tickers provided")

            # Clean tickers
            tickers = [t.upper().strip() for t in tickers]

            # Check memory for similar analyses
            context = await self.memory.get_context(tickers, analysis_type)

            # Execute the crew
            start_time = datetime.now()

            result = await self.crew.kickoff({
                "tickers": tickers,
                "analysis_type": analysis_type,
                "context": context
            })

            processing_time = (datetime.now() - start_time).total_seconds()

            # Add performance metrics
            result['processing_time'] = processing_time
            result['performance_score'] = self._calculate_performance_score(result)
            result['analysis_id'] = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Store results in memory
            store_result = await self.memory.store_analysis(result)
            if store_result.get('status') == 'success':
                logger.info(f"Analysis stored with ID: {store_result.get('analysis_id')}")

            # Update analysis count
            self.analysis_count += 1

            # Trigger evolution if conditions met
            if self.analysis_count % settings.EVOLUTION_TRIGGER == 0 and not self.is_evolving:
                logger.info(f"Triggering evolution after {self.analysis_count} analyses")
                asyncio.create_task(self._trigger_evolution())

            logger.info(f"Analysis completed in {processing_time:.2f} seconds")
            return result

        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)

            # Return error result
            return {
                'status': 'error',
                'error': str(e),
                'tickers': tickers,
                'analysis_type': analysis_type,
                'timestamp': datetime.now().isoformat(),
                'suggested_fixes': [
                    'Check network connectivity',
                    'Verify API keys are valid',
                    'Ensure all services are running'
                ]
            }

    def _calculate_performance_score(self, result: dict) -> float:
        """Calculate a performance score for the analysis"""
        try:
            score = 0.5  # Base score

            # Add points for completeness
            if result.get('recommendation'):
                score += 0.2

            if result.get('confidence') and result.get('confidence') > 0.7:
                score += 0.1

            if result.get('analysis') and len(result['analysis']) > 100:
                score += 0.1

            if result.get('visualizations'):
                score += 0.1

            # Deduct points for errors
            if result.get('status') == 'error':
                score -= 0.3

            # Ensure score is between 0 and 1
            return max(0.0, min(1.0, score))

        except:
            return 0.5

    async def _trigger_evolution(self):
        """Trigger self-evolution process"""
        if self.is_evolving:
            logger.warning("Evolution already in progress")
            return

        self.is_evolving = True
        try:
            logger.info("Starting evolution process")

            # Get evolution patterns
            patterns = await self.memory.get_evolution_patterns(
                min_score=settings.PERFORMANCE_THRESHOLD
            )

            if patterns:
                logger.info(f"Found {len(patterns)} successful patterns for evolution")

                # Here you would implement the evolution logic
                # For now, just log the patterns
                for pattern in patterns:
                    logger.info(f"Pattern: {pattern.get('type')}, "
                                f"Avg Score: {pattern.get('avg_score'):.2f}, "
                                f"Count: {pattern.get('success_count')}")

            # Simulate evolution delay
            await asyncio.sleep(10)

            logger.info("Evolution process completed")

        except Exception as e:
            logger.error(f"Evolution failed: {e}", exc_info=True)

        finally:
            self.is_evolving = False

    async def get_system_status(self) -> dict:
        """Get current system status"""
        return {
            'status': 'running',
            'analysis_count': self.analysis_count,
            'is_evolving': self.is_evolving,
            'memory_available': True,  # Would check actual memory
            'services': {
                'mcp_servers': True,  # Would check actual services
                'database': True,
                'llm': True
            },
            'timestamp': datetime.now().isoformat()
        }

    async def cleanup(self):
        """Clean up system resources"""
        logger.info("Cleaning up system resources")

        # Clean up old analyses
        cleanup_result = await self.memory.cleanup_old_analyses(days_old=30)
        logger.info(f"Cleanup result: {cleanup_result}")

    def shutdown(self):
        """Shutdown the system gracefully"""
        logger.info("Shutting down AlphaForge System")
        # Add any cleanup logic here


async def main():
    """Main entry point"""
    logger.info("Starting AlphaForge Financial Analysis System")

    try:
        # Initialize system
        system = AlphaForgeSystem()

        # Check system status
        status = await system.get_system_status()
        logger.info(f"System status: {status}")

        # Example: Analyze Magnificent 7
        result = await system.analyze_stocks([
            "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA"
        ])

        logger.info(f"Analysis result status: {result.get('status')}")

        # Clean up before exit
        await system.cleanup()

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1

    logger.info("AlphaForge System shutdown complete")
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
