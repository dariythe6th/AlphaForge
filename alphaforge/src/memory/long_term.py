# src/memory/long_term.py
import redis
from neo4j import GraphDatabase
import chromadb
from chromadb.config import Settings as ChromaSettings
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LongTermMemory:
    def __init__(self, redis_url: str = "redis://localhost:6379",
                 neo4j_uri: str = None, neo4j_user: str = "neo4j",
                 neo4j_password: str = "", chroma_path: str = "./memory_db"):

        # Initialize thread pool
        self.executor = ThreadPoolExecutor(max_workers=2)

        # Initialize Redis
        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            self.redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None

        # Initialize ChromaDB for analysis history
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=chroma_path,
                settings=ChromaSettings(anonymized_telemetry=False)
            )
            self.analysis_memory = self.chroma_client.get_or_create_collection(
                name="analysis_history",
                metadata={"description": "Historical financial analyses"}
            )
            logger.info(f"ChromaDB memory initialized at {chroma_path}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB memory: {e}")
            raise

        # Initialize Neo4j for performance metrics
        if neo4j_uri:
            try:
                self.neo4j_driver = GraphDatabase.driver(
                    neo4j_uri,
                    auth=(neo4j_user, neo4j_password)
                )
                # Test connection
                with self.neo4j_driver.session() as session:
                    session.run("RETURN 1")
                logger.info("Neo4j connection established for memory")
            except Exception as e:
                logger.error(f"Failed to connect to Neo4j: {e}")
                self.neo4j_driver = None
        else:
            self.neo4j_driver = None

        # Initialize in-memory cache for frequent queries
        self.local_cache = {}
        self.cache_ttl = 3600  # 1 hour

    async def store_analysis(self, analysis_result: Dict[str, Any]):
        """Store analysis results in long-term memory asynchronously"""
        try:
            logger.info("Storing analysis in long-term memory")

            def store_sync(result):
                # Prepare analysis data
                analysis_id = result.get('id', f"analysis_{datetime.now().timestamp()}")
                analysis_data = json.dumps(result)
                tickers = result.get('tickers', [])

                # Store in ChromaDB
                self.analysis_memory.add(
                    ids=[analysis_id],
                    documents=[analysis_data],
                    metadatas=[{
                        'tickers': tickers,
                        'timestamp': datetime.now().isoformat(),
                        'analysis_type': result.get('analysis_type', 'comprehensive'),
                        'performance_score': float(result.get('performance_score', 0)),
                        'agent': result.get('agent', 'unknown')
                    }]
                )

                # Store in Redis for quick access
                if self.redis_client:
                    cache_key = f"analysis:{analysis_id}"
                    self.redis_client.setex(
                        cache_key,
                        self.cache_ttl,
                        analysis_data
                    )

                # Store performance metrics in Neo4j
                if self.neo4j_driver:
                    self._store_performance_metrics_sync(result, analysis_id)

                return analysis_id

            # Run in thread pool
            stored_id = await asyncio.get_event_loop().run_in_executor(
                self.executor, store_sync, analysis_result
            )

            return {"status": "success", "analysis_id": stored_id}

        except Exception as e:
            logger.error(f"Error storing analysis: {e}")
            return {"status": "error", "error": str(e)}

    async def get_context(self, tickers: List[str], analysis_type: str) -> List[Dict[str, Any]]:
        """Retrieve relevant context from past analyses asynchronously"""
        try:
            logger.info(f"Retrieving context for {tickers}, type: {analysis_type}")

            def retrieve_context_sync():
                # Try Redis cache first
                if self.redis_client:
                    cache_key = f"context:{':'.join(sorted(tickers))}:{analysis_type}"
                    cached = self.redis_client.get(cache_key)
                    if cached:
                        return json.loads(cached)

                # Query ChromaDB
                query_text = f"Analysis of {', '.join(tickers)} for {analysis_type}"

                results = self.analysis_memory.query(
                    query_texts=[query_text],
                    n_results=5,
                    where={"analysis_type": analysis_type},
                    include=["documents", "metadatas", "distances"]
                )

                context = []
                if results['documents'] and results['documents'][0]:
                    for doc, metadata, distance in zip(
                            results['documents'][0],
                            results['metadatas'][0],
                            results['distances'][0]
                    ):
                        try:
                            analysis_data = json.loads(doc)
                            context.append({
                                'previous_analysis': analysis_data,
                                'metadata': metadata,
                                'relevance_score': 1 - distance if distance <= 1 else 0
                            })
                        except json.JSONDecodeError:
                            continue

                # Cache in Redis
                if self.redis_client and context:
                    cache_key = f"context:{':'.join(sorted(tickers))}:{analysis_type}"
                    self.redis_client.setex(
                        cache_key,
                        300,  # 5 minutes TTL for context
                        json.dumps(context)
                    )

                return context

            # Run in thread pool
            context = await asyncio.get_event_loop().run_in_executor(
                self.executor, retrieve_context_sync
            )

            logger.info(f"Retrieved {len(context)} context items")
            return context

        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return []

    def _store_performance_metrics_sync(self, analysis_result: Dict[str, Any], analysis_id: str):
        """Store performance metrics for evolutionary learning (synchronous)"""
        try:
            with self.neo4j_driver.session() as session:
                session.run("""
                    MERGE (a:Analysis {id: $id})
                    SET a.timestamp = $timestamp,
                        a.analysis_type = $analysis_type,
                        a.performance_score = $performance_score,
                        a.agent_version = $agent_version

                    WITH a
                    UNWIND $tickers AS ticker
                    MERGE (c:Company {symbol: ticker})
                    MERGE (a)-[r:ANALYZES {timestamp: $rel_timestamp}]->(c)
                    SET r.recommendation = $recommendation,
                        r.confidence = $confidence,
                        r.target_price = $target_price

                    RETURN a.id
                """,
                            id=analysis_id,
                            timestamp=datetime.now().isoformat(),
                            analysis_type=analysis_result.get('analysis_type', 'comprehensive'),
                            performance_score=float(analysis_result.get('performance_score', 0)),
                            agent_version=analysis_result.get('agent_version', '1.0'),
                            tickers=analysis_result.get('tickers', []),
                            rel_timestamp=datetime.now().isoformat(),
                            recommendation=analysis_result.get('recommendation', 'HOLD'),
                            confidence=float(analysis_result.get('confidence', 0)),
                            target_price=float(analysis_result.get('target_price', 0))
                            )

            logger.debug(f"Stored performance metrics for analysis {analysis_id}")

        except Exception as e:
            logger.error(f"Error storing performance metrics: {e}")

    async def get_evolution_patterns(self, min_score: float = 0.8, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve patterns for strategy evolution asynchronously"""
        try:
            logger.info(f"Retrieving evolution patterns with min_score={min_score}")

            def retrieve_patterns_sync():
                if not self.neo4j_driver:
                    return []

                with self.neo4j_driver.session() as session:
                    result = session.run("""
                        MATCH (a:Analysis)
                        WHERE a.performance_score >= $min_score
                        RETURN a.analysis_type as type, 
                               a.agent_version as agent_version,
                               COUNT(*) as success_count,
                               AVG(a.performance_score) as avg_score,
                               COLLECT(DISTINCT a.id)[0..5] as sample_ids
                        ORDER BY avg_score DESC
                        LIMIT $limit
                    """, min_score=min_score, limit=limit)

                    patterns = []
                    for record in result:
                        patterns.append({
                            'type': record['type'],
                            'agent_version': record['agent_version'],
                            'success_count': record['success_count'],
                            'avg_score': float(record['avg_score']),
                            'sample_ids': record['sample_ids']
                        })

                    return patterns

            # Run in thread pool
            patterns = await asyncio.get_event_loop().run_in_executor(
                self.executor, retrieve_patterns_sync
            )

            logger.info(f"Retrieved {len(patterns)} evolution patterns")
            return patterns

        except Exception as e:
            logger.error(f"Error retrieving evolution patterns: {e}")
            return []

    async def cleanup_old_analyses(self, days_old: int = 30):
        """Clean up analyses older than specified days"""
        try:
            def cleanup_sync():
                cutoff_date = (datetime.now() - timedelta(days=days_old)).isoformat()

                # Remove from ChromaDB
                # Note: ChromaDB doesn't have built-in TTL, so we need to query and delete
                # This is a simplified version
                if self.redis_client:
                    # Clean Redis cache
                    pattern = "analysis:*"
                    keys = self.redis_client.keys(pattern)
                    for key in keys:
                        self.redis_client.delete(key)

                logger.info(f"Cleaned up analyses older than {days_old} days")
                return {"status": "success", "cleaned_up": True}

            # Run in thread pool
            return await asyncio.get_event_loop().run_in_executor(
                self.executor, cleanup_sync
            )

        except Exception as e:
            logger.error(f"Error cleaning up old analyses: {e}")
            return {"status": "error", "error": str(e)}