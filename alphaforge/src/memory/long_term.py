# src/memory/long_term.py
import redis
from neo4j import GraphDatabase
import chromadb
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List
import asyncio

class LongTermMemory:
    def __init__(self):
        # Short-term memory (session-based)
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        
        # Long-term vector memory
        self.chroma_client = chromadb.PersistentClient(path="./memory_db")
        self.analysis_memory = self.chroma_client.get_or_create_collection("analysis_history")
        
        # Performance database
        self.neo4j_driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
    
    async def store_analysis(self, analysis_result: Dict[str, Any]):
        """Store analysis results in long-term memory"""
        # Store in vector database
        self.analysis_memory.add(
            documents=[json.dumps(analysis_result)],
            metadatas=[{
                'tickers': analysis_result.get('tickers', []),
                'timestamp': datetime.now().isoformat(),
                'analysis_type': analysis_result.get('analysis_type', 'comprehensive'),
                'performance_score': analysis_result.get('performance_score', 0)
            }],
            ids=[f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"]
        )
        
        # Store performance metrics in graph database
        await self._store_performance_metrics(analysis_result)
    
    async def get_context(self, tickers: List[str], analysis_type: str) -> List[Dict[str, Any]]:
        """Retrieve relevant context from past analyses"""
        # Semantic search for similar analyses
        query = f"Analysis of {', '.join(tickers)} for {analysis_type}"
        
        results = self.analysis_memory.query(
            query_texts=[query],
            n_results=5,
            where={"analysis_type": analysis_type}  # Filter by type
        )
        
        context = []
        for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
            context.append({
                'previous_analysis': json.loads(doc),
                'metadata': metadata,
                'relevance_score': results['distances'][0][0]  # Convert distance to score
            })
        
        return context
    
    async def _store_performance_metrics(self, analysis_result: Dict[str, Any]):
        """Store performance metrics for evolutionary learning"""
        with self.neo4j_driver.session() as session:
            session.run("""
                MERGE (a:Analysis {id: $id})
                SET a.timestamp = $timestamp,
                    a.analysis_type = $analysis_type,
                    a.performance_score = $performance_score
                
                WITH a
                UNWIND $tickers AS ticker
                MERGE (c:Company {symbol: ticker})
                MERGE (a)-[r:ANALYZES]->(c)
                SET r.recommendation = $recommendation,
                    r.confidence = $confidence
            """, 
            id=analysis_result.get('id'),
            timestamp=datetime.now().isoformat(),
            analysis_type=analysis_result.get('analysis_type'),
            performance_score=analysis_result.get('performance_score', 0),
            tickers=analysis_result.get('tickers', []),
            recommendation=analysis_result.get('recommendation'),
            confidence=analysis_result.get('confidence', 0)
            )
    
    async def get_evolution_patterns(self) -> List[Dict[str, Any]]:
        """Retrieve patterns for strategy evolution"""
        with self.neo4j_driver.session() as session:
            result = session.run("""
                MATCH (a:Analysis)
                WHERE a.performance_score > 0.8
                RETURN a.analysis_type as type, 
                       COUNT(*) as success_count,
                       AVG(a.performance_score) as avg_score
                ORDER BY avg_score DESC
                LIMIT 10
            """)
            
            return [dict(record) for record in result]