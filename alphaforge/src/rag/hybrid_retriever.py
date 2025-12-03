# src/rag/hybrid_retriever.py
import chromadb
from chromadb.config import Settings as ChromaSettings
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridRetriever:
    def __init__(self, chroma_path: str = "./chroma_db", neo4j_uri: str = None,
                 neo4j_user: str = "neo4j", neo4j_password: str = ""):

        # Initialize thread pool
        self.executor = ThreadPoolExecutor(max_workers=2)

        # Initialize ChromaDB
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=chroma_path,
                settings=ChromaSettings(anonymized_telemetry=False)
            )
            self.collection = self.chroma_client.get_or_create_collection(
                name="financial_knowledge",
                metadata={"description": "Financial analysis knowledge base"}
            )
            logger.info(f"ChromaDB initialized at {chroma_path}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

        # Initialize Neo4j
        if neo4j_uri:
            try:
                self.neo4j_driver = GraphDatabase.driver(
                    neo4j_uri,
                    auth=(neo4j_user, neo4j_password)
                )
                # Test connection
                with self.neo4j_driver.session() as session:
                    session.run("RETURN 1")
                logger.info("Neo4j connection established")
            except Exception as e:
                logger.error(f"Failed to connect to Neo4j: {e}")
                self.neo4j_driver = None
        else:
            self.neo4j_driver = None

        # Initialize embeddings model
        try:
            self.embeddings = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Embeddings model loaded")
        except Exception as e:
            logger.error(f"Failed to load embeddings model: {e}")
            raise

    async def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to both vector store and knowledge graph asynchronously"""
        try:
            logger.info(f"Adding {len(documents)} documents")

            def process_documents_sync(docs):
                # Prepare data for ChromaDB
                ids = []
                texts = []
                metadatas = []
                embeddings_list = []

                for i, doc in enumerate(docs):
                    content = doc.get('content', '')
                    metadata = doc.get('metadata', {})

                    if not content:
                        continue

                    # Generate embedding
                    embedding = self.embeddings.encode(content).tolist()

                    # Prepare for ChromaDB
                    ids.append(f"doc_{i}_{datetime.now().timestamp()}")
                    texts.append(content)
                    metadatas.append({
                        **metadata,
                        'added_at': datetime.now().isoformat(),
                        'source_type': doc.get('type', 'unknown')
                    })
                    embeddings_list.append(embedding)

                # Add to ChromaDB
                if texts:
                    self.collection.add(
                        ids=ids,
                        documents=texts,
                        metadatas=metadatas,
                        embeddings=embeddings_list
                    )
                    logger.info(f"Added {len(texts)} documents to ChromaDB")

                # Add to Neo4j if available
                if self.neo4j_driver and docs:
                    self._add_to_knowledge_graph_sync(docs)

                return len(texts)

            # Run in thread pool
            added_count = await asyncio.get_event_loop().run_in_executor(
                self.executor, process_documents_sync, documents
            )

            return {"status": "success", "added_count": added_count}

        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return {"status": "error", "error": str(e)}

    async def retrieve(self, query: str, top_k: int = 5,
                       strategy: str = "hybrid") -> List[Dict[str, Any]]:
        """Hybrid retrieval with dynamic strategy selection"""
        try:
            logger.info(f"Retrieving with strategy: {strategy}, top_k: {top_k}")

            if strategy == "factual" and self.neo4j_driver:
                results = await self._kg_retrieval(query, top_k)
            elif strategy == "conceptual":
                results = await self._vector_retrieval(query, top_k)
            else:  # hybrid
                # Run both retrievals in parallel
                vector_task = self._vector_retrieval(query, top_k)
                kg_task = self._kg_retrieval(query, top_k) if self.neo4j_driver else []

                vector_results, kg_results = await asyncio.gather(vector_task, kg_task)

                # Combine and re-rank
                combined = vector_results + kg_results
                results = self._rerank_results(query, combined, top_k)

            logger.info(f"Retrieved {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Error in retrieval: {e}")
            return []

    async def _vector_retrieval(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Vector-based semantic search"""
        try:
            def retrieve_sync():
                # Generate query embedding
                query_embedding = self.embeddings.encode(query).tolist()

                # Query ChromaDB
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"]
                )

                retrieved = []
                if results['documents'] and results['documents'][0]:
                    for i, (doc, metadata, distance) in enumerate(zip(
                            results['documents'][0],
                            results['metadatas'][0],
                            results['distances'][0]
                    )):
                        # Convert distance to similarity score (cosine distance to similarity)
                        similarity = 1 - distance if distance <= 1 else 0

                        retrieved.append({
                            'content': doc,
                            'metadata': metadata,
                            'score': float(similarity),
                            'type': 'vector',
                            'id': results['ids'][0][i] if results['ids'][0] else f"vec_{i}"
                        })

                return retrieved

            # Run in thread pool
            return await asyncio.get_event_loop().run_in_executor(
                self.executor, retrieve_sync
            )

        except Exception as e:
            logger.error(f"Error in vector retrieval: {e}")
            return []

    async def _kg_retrieval(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Knowledge graph pattern matching"""
        try:
            def retrieve_kg_sync():
                if not self.neo4j_driver:
                    return []

                with self.neo4j_driver.session() as session:
                    # Simple pattern matching for now
                    result = session.run("""
                        MATCH (c:Company)-[r]->(m:Metric)
                        WHERE c.name CONTAINS $query OR c.symbol CONTAINS $query 
                           OR m.name CONTAINS $query OR m.description CONTAINS $query
                        RETURN c.name as company, c.symbol as symbol,
                               type(r) as relationship, 
                               m.name as metric, m.value as value,
                               m.unit as unit, m.timestamp as timestamp
                        LIMIT $limit
                    """, query=query, limit=top_k)

                    retrieved = []
                    for record in result:
                        content = f"{record['company']} ({record['symbol']}) {record['relationship']} {record['metric']}: {record['value']} {record.get('unit', '')}"

                        retrieved.append({
                            'content': content,
                            'metadata': {
                                'source': 'knowledge_graph',
                                'type': 'fact',
                                'company': record['company'],
                                'symbol': record['symbol'],
                                'metric': record['metric'],
                                'timestamp': record['timestamp']
                            },
                            'score': 0.9,  # High confidence for KG facts
                            'type': 'knowledge_graph'
                        })

                    return retrieved

            # Run in thread pool
            return await asyncio.get_event_loop().run_in_executor(
                self.executor, retrieve_kg_sync
            )

        except Exception as e:
            logger.error(f"Error in KG retrieval: {e}")
            return []

    def _add_to_knowledge_graph_sync(self, documents: List[Dict[str, Any]]):
        """Extract entities and relationships for knowledge graph (synchronous)"""
        # Simplified implementation - in production, use NER models
        try:
            for doc in documents:
                content = doc.get('content', '')
                metadata = doc.get('metadata', {})

                # Simple extraction for demonstration
                # In production, use proper NLP pipelines
                if 'AAPL' in content or 'Apple' in content:
                    with self.neo4j_driver.session() as session:
                        session.run("""
                            MERGE (c:Company {symbol: 'AAPL', name: 'Apple Inc.'})
                            MERGE (m:Metric {name: $metric_name, description: $desc})
                            MERGE (c)-[:REPORTS {value: $value, timestamp: $ts}]->(m)
                        """,
                                    metric_name=metadata.get('metric', 'unknown'),
                                    desc=content[:100],
                                    value=metadata.get('value', 0),
                                    ts=metadata.get('timestamp', datetime.now().isoformat())
                                    )

        except Exception as e:
            logger.warning(f"Failed to add to knowledge graph: {e}")

    def _rerank_results(self, query: str, results: List[Dict[str, Any]],
                        top_k: int) -> List[Dict[str, Any]]:
        """Simple reranking based on type and score"""
        if not results:
            return []

        # Apply boosting
        for result in results:
            # Boost KG results for factual queries
            if result['type'] == 'knowledge_graph':
                factual_keywords = ['what', 'when', 'who', 'how much', 'price', 'revenue', 'earnings']
                if any(keyword in query.lower() for keyword in factual_keywords):
                    result['score'] *= 1.2

            # Penalize very short documents
            if len(result.get('content', '')) < 50:
                result['score'] *= 0.7

        # Sort by score and limit
        sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
        return sorted_results[:top_k]