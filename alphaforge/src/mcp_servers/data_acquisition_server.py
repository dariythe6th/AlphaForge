# src/mcp_servers/data_acquisition_server.py
from mcp.server import MCPServer
from mcp.server.models import InitializationOptions
import mcp.types as types
import yfinance as yf
from newsapi import NewsApiClient
from sec_edgar_downloader import Downloader
import pytesseract
from PIL import Image
import pandas as pd
import aiohttp
import asyncio
from typing import Dict, Any, List
import json

class DataAcquisitionServer(MCPServer):
    def __init__(self):
        super().__init__()
        self.newsapi = NewsApiClient(api_key='your_newsapi_key')
        self.sec_downloader = Downloader()
        self.cache = {}
        
    async def handle_stock_prices(self, ticker: str, period: str = "1y") -> Dict[str, Any]:
        """Fetch and normalize stock price data"""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            
            # Normalize data
            normalized = {
                'ticker': ticker,
                'prices': hist[['Open', 'High', 'Low', 'Close', 'Volume']].reset_index().to_dict('records'),
                'info': {
                    'company_name': stock.info.get('longName', ''),
                    'sector': stock.info.get('sector', ''),
                    'market_cap': stock.info.get('marketCap', 0)
                }
            }
            return normalized
        except Exception as e:
            return {'error': str(e)}
    
    async def handle_news_sentiment(self, ticker: str, days: int = 30) -> Dict[str, Any]:
        """Fetch news and perform sentiment analysis"""
        try:
            from transformers import pipeline
            sentiment_analyzer = pipeline("sentiment-analysis")
            
            news = self.newsapi.get_everything(
                q=ticker,
                language='en',
                sort_by='relevancy',
                from_param=pd.Timestamp.now() - pd.Timedelta(days=days)
            )
            
            articles = []
            for article in news['articles'][:50]:  # Limit to top 50
                sentiment = sentiment_analyzer(article['title'])[0]
                articles.append({
                    'title': article['title'],
                    'description': article['description'],
                    'published_at': article['publishedAt'],
                    'sentiment': sentiment['label'],
                    'score': sentiment['score']
                })
            
            # Calculate aggregate sentiment
            positive = len([a for a in articles if a['sentiment'] == 'POSITIVE'])
            total = len(articles)
            sentiment_score = positive / total if total > 0 else 0.5
            
            return {
                'ticker': ticker,
                'articles': articles,
                'sentiment_score': sentiment_score,
                'article_count': total
            }
        except Exception as e:
            return {'error': str(e)}
    
    async def handle_sec_filings(self, ticker: str, filing_type: str = "10-K") -> Dict[str, Any]:
        """Download and parse SEC filings"""
        try:
            # Download latest filing
            self.sec_downloader.get(filing_type, ticker, limit=1)
            
            # Parse filing text (simplified)
            filing_path = f"sec_filings/{ticker}/{filing_type}/"
            # Implementation for parsing PDF and extracting financial tables
            
            return {
                'ticker': ticker,
                'filing_type': filing_type,
                'filing_date': '2024-01-01',  # Actual implementation would extract this
                'key_metrics': {}  # Extracted financial metrics
            }
        except Exception as e:
            return {'error': str(e)}
    
    async def handle_image_analysis(self, image_url: str) -> Dict[str, Any]:
        """Perform OCR and analysis on financial images/charts"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url) as response:
                    image_data = await response.read()
            
            # Perform OCR
            image = Image.open(io.BytesIO(image_data))
            text = pytesseract.image_to_string(image)
            
            return {
                'extracted_text': text,
                'image_size': image.size,
                'analysis': 'Chart analysis placeholder'  # Actual chart analysis would go here
            }
        except Exception as e:
            return {'error': str(e)}
    
    async def handle_list_tools(self) -> List[types.Tool]:
        return [
            types.Tool(
                name="get_stock_prices",
                description="Fetch historical stock price data",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string"},
                        "period": {"type": "string", "default": "1y"}
                    },
                    "required": ["ticker"]
                }
            ),
            types.Tool(
                name="get_news_sentiment",
                description="Fetch news articles and perform sentiment analysis",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string"},
                        "days": {"type": "integer", "default": 30}
                    },
                    "required": ["ticker"]
                }
            )
        ]
    
    async def handle_call_tool(self, name: str, arguments: dict) -> List[types.TextContent]:
        if name == "get_stock_prices":
            result = await self.handle_stock_prices(**arguments)
        elif name == "get_news_sentiment":
            result = await self.handle_news_sentiment(**arguments)
        elif name == "get_sec_filings":
            result = await self.handle_sec_filings(**arguments)
        elif name == "analyze_image":
            result = await self.handle_image_analysis(**arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")
        
        return [types.TextContent(type="text", text=json.dumps(result))]