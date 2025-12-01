# tests/test_system.py
import pytest
import asyncio
from src.agents.orchestrator import AlphaForgeCrew
from src.memory.long_term import LongTermMemory

class TestAlphaForge:
    @pytest.fixture
    async def system(self):
        return AlphaForgeCrew()
    
    @pytest.mark.asyncio
    async def test_stock_analysis(self, system):
        """Test basic stock analysis functionality"""
        result = await system.kickoff({
            "tickers": ["AAPL"],
            "analysis_type": "comprehensive"
        })
        
        assert result is not None
        assert "recommendation" in result
        assert "analysis" in result
        assert "visualizations" in result
    
    @pytest.mark.asyncio 
    async def test_multi_modal_data(self, system):
        """Test multi-modal data processing"""
        # This would test that all 4 data types are processed
        pass
    
    @pytest.mark.asyncio
    async def test_performance_benchmark(self, system):
        """Test system performance meets requirements"""
        import time
        
        start_time = time.time()
        result = await system.kickoff({
            "tickers": ["NVDA", "MSFT"],
            "analysis_type": "comprehensive"
        })
        end_time = time.time()
        
        # Should complete within 10 minutes
        assert (end_time - start_time) < 600
        
        # Should have all required components
        assert len(result.get("tickers", [])) == 2

if __name__ == "__main__":
    pytest.main()