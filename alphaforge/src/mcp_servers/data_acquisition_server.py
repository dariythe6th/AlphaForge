# src/mcp_servers/data_acquisition_server.py
import os
import io
import json
import logging
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor
import asyncio

from mcp.server import MCPServer
from mcp.server.models import InitializationOptions
import mcp.types as types

import yfinance as yf
import pandas as pd
from newsapi import NewsApiClient
from sec_edgar_downloader import Downloader
import pytesseract
from PIL import Image
import aiohttp
from transformers import pipeline
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataAcquisitionServer(MCPServer):
    def __init__(self):
        super().__init__()

        # Initialize with environment variables
        self.newsapi = NewsApiClient(api_key=os.getenv("NEWSAPI_KEY"))
        self.sec_downloader = Downloader()
        self.cache = {}

        # Create thread pool for blocking operations
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Initialize models once
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )

    async def handle_stock_prices(self, ticker: str, period: str = "1y") -> Dict[str, Any]:
        """Fetch and normalize stock price data asynchronously"""
        try:
            logger.info(f"Fetching stock prices for {ticker} over {period}")

            def fetch_sync():
                stock = yf.Ticker(ticker)
                hist = stock.history(period=period)
                return stock, hist

            # Run blocking call in thread pool
            stock, hist = await asyncio.get_event_loop().run_in_executor(
                self.executor, fetch_sync
            )

            # Ensure we have data
            if hist.empty:
                return {"error": f"No data available for {ticker}"}

            # Normalize data
            normalized = {
                'ticker': ticker,
                'prices': hist[['Open', 'High', 'Low', 'Close', 'Volume']]
                .reset_index()
                .to_dict('records'),
                'info': {
                    'company_name': stock.info.get('longName', ''),
                    'sector': stock.info.get('sector', ''),
                    'market_cap': stock.info.get('marketCap', 0),
                    'currency': stock.info.get('currency', 'USD')
                },
                'period': period,
                'last_updated': datetime.now().isoformat()
            }
            return normalized

        except Exception as e:
            logger.error(f"Error fetching stock prices for {ticker}: {e}")
            return {'error': str(e), 'ticker': ticker}

    async def handle_news_sentiment(self, ticker: str, days: int = 30) -> Dict[str, Any]:
        """Fetch news and perform sentiment analysis asynchronously"""
        try:
            logger.info(f"Fetching news sentiment for {ticker} over {days} days")

            # Calculate date
            since_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

            # Fetch news
            news = self.newsapi.get_everything(
                q=ticker,
                language='en',
                sort_by='relevancy',
                from_param=since_date,
                page_size=50
            )

            articles = []

            # Analyze sentiment in batches
            def analyze_batch(batch):
                results = []
                for article in batch:
                    text = article.get('title', '')[:512]
                    if not text:
                        results.append({'label': 'NEUTRAL', 'score': 0.5})
                    else:
                        sentiment_result = self.sentiment_analyzer(text)[0]
                        results.append(sentiment_result)
                return results

            # Process in batches for efficiency
            batch_size = 10
            news_articles = news.get('articles', [])

            for i in range(0, len(news_articles), batch_size):
                batch = news_articles[i:i + batch_size]
                batch_results = await asyncio.get_event_loop().run_in_executor(
                    self.executor, analyze_batch, batch
                )

                for article, sentiment_result in zip(batch, batch_results):
                    label = sentiment_result['label']
                    # Convert to consistent format
                    sentiment_label = 'POSITIVE' if 'POSITIVE' in label.upper() else 'NEGATIVE'

                    articles.append({
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'published_at': article.get('publishedAt'),
                        'sentiment': sentiment_label,
                        'score': sentiment_result['score']
                    })

            # Calculate aggregate sentiment
            if articles:
                positive = len([a for a in articles if a['sentiment'] == 'POSITIVE'])
                total = len(articles)
                sentiment_score = positive / total if total > 0 else 0.5
            else:
                sentiment_score = 0.5

            return {
                'ticker': ticker,
                'articles': articles[:20],  # Return top 20 for brevity
                'sentiment_score': sentiment_score,
                'article_count': len(articles),
                'date_range': f"{since_date} to {datetime.now().strftime('%Y-%m-%d')}"
            }

        except Exception as e:
            logger.error(f"Error fetching news sentiment for {ticker}: {e}")
            return {'error': str(e), 'ticker': ticker}

    async def handle_sec_filings(self, ticker: str, filing_type: str = "10-K") -> Dict[str, Any]:
        """Download and parse SEC filings asynchronously"""
        try:
            logger.info(f"Fetching SEC filings for {ticker} - {filing_type}")

            def download_filing():
                # Limit to 1 filing to avoid rate limiting
                self.sec_downloader.get(filing_type, ticker, limit=1)
                return True

            # Run in thread pool
            success = await asyncio.get_event_loop().run_in_executor(
                self.executor, download_filing
            )

            if success:
                return {
                    'ticker': ticker,
                    'filing_type': filing_type,
                    'status': 'downloaded',
                    'path': f"sec_filings/{ticker}/{filing_type}/",
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {'error': 'Failed to download filing', 'ticker': ticker}

        except Exception as e:
            logger.error(f"Error fetching SEC filings for {ticker}: {e}")
            return {'error': str(e), 'ticker': ticker}

    async def handle_image_analysis(self, image_url: str) -> Dict[str, Any]:
        """Perform OCR and analysis on financial images/charts asynchronously"""
        try:
            logger.info(f"Analyzing image from {image_url}")

            # Download image
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url, timeout=30) as response:
                    if response.status != 200:
                        return {'error': f'Failed to download image: {response.status}'}
                    image_data = await response.read()

            # Perform OCR in thread pool
            def perform_ocr(img_data):
                try:
                    image = Image.open(io.BytesIO(img_data)).convert("RGB")
                    text = pytesseract.image_to_string(image)
                    return text, image.size
                except Exception as e:
                    logger.error(f"OCR error: {e}")
                    return "", (0, 0)

            text, size = await asyncio.get_event_loop().run_in_executor(
                self.executor, perform_ocr, image_data
            )

            return {
                'extracted_text': text,
                'image_size': size,
                'url': image_url,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error analyzing image {image_url}: {e}")
            return {'error': str(e), 'url': image_url}

    async def handle_list_tools(self) -> List[types.Tool]:
        """List available tools"""
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
            ),
            types.Tool(
                name="get_sec_filings",
                description="Download SEC filings",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string"},
                        "filing_type": {"type": "string", "default": "10-K"}
                    },
                    "required": ["ticker"]
                }
            ),
            types.Tool(
                name="analyze_image",
                description="Perform OCR and analysis on financial images/charts",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "image_url": {"type": "string"}
                    },
                    "required": ["image_url"]
                }
            )
        ]

    async def handle_call_tool(self, name: str, arguments: dict) -> List[types.TextContent]:
        """Handle tool calls"""
        try:
            # Validate input
            if not arguments:
                raise ValueError("No arguments provided")

            # Route to appropriate handler
            if name == "get_stock_prices":
                ticker = arguments.get('ticker', '').upper()
                if not ticker or len(ticker) > 10:
                    raise ValueError("Invalid ticker symbol")
                period = arguments.get('period', '1y')
                result = await self.handle_stock_prices(ticker, period)

            elif name == "get_news_sentiment":
                ticker = arguments.get('ticker', '').upper()
                if not ticker or len(ticker) > 10:
                    raise ValueError("Invalid ticker symbol")
                days = min(max(int(arguments.get('days', 30)), 1), 365)  # Limit 1-365 days
                result = await self.handle_news_sentiment(ticker, days)

            elif name == "get_sec_filings":
                ticker = arguments.get('ticker', '').upper()
                if not ticker or len(ticker) > 10:
                    raise ValueError("Invalid ticker symbol")
                filing_type = arguments.get('filing_type', '10-K')
                result = await self.handle_sec_filings(ticker, filing_type)

            elif name == "analyze_image":
                image_url = arguments.get('image_url', '')
                if not image_url.startswith(('http://', 'https://')):
                    raise ValueError("Invalid image URL")
                result = await self.handle_image_analysis(image_url)

            else:
                raise ValueError(f"Unknown tool: {name}")

            return [types.TextContent(type="text", text=json.dumps(result))]

        except Exception as e:
            logger.error(f"Error handling tool {name}: {e}")
            error_result = {'error': str(e), 'tool': name}
            return [types.TextContent(type="text", text=json.dumps(error_result))]