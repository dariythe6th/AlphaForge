# src/rag/hybrid_retriever.py
import chromadb
from neo4j import GraphDatabase
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List, Dict, Any
import asyncio

class HybridRetriever:
    def __init__(self):
        # Vector store
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.chroma_client.get_or_create_collection("financial_knowledge")
        
        # Knowledge graph
        self.neo4j_driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
        
        # Embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
    
    async def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to both vector store and knowledge graph"""
        for doc in documents:
            # Split text
            chunks = self.text_splitter.split_text(doc['content'])
            
            # Add to vector store
            embeddings = self.embeddings.embed_documents(chunks)
            self.collection.add(
                embeddings=embeddings,
                documents=chunks,
                metadatas=[doc['metadata'] for _ in chunks],
                ids=[f"{doc['id']}_{i}" for i in range(len(chunks))]
            )
            
            # Add to knowledge graph
            await self._add_to_knowledge_graph(doc)
    
    async def retrieve(self, query: str, top_k: int = 5, strategy: str = "hybrid") -> List[Dict[str, Any]]:
        """Hybrid retrieval with dynamic strategy selection"""
        
        if strategy == "factual":
            return await self._kg_retrieval(query, top_k)
        elif strategy == "conceptual":
            return await self._vector_retrieval(query, top_k)
        else:  # hybrid
            vector_results = await self._vector_retrieval(query, top_k)
            kg_results = await self._kg_retrieval(query, top_k)
            
            # Combine and re-rank
            combined = self._rerank_results(query, vector_results + kg_results)
            return combined[:top_k]
    
    async def _vector_retrieval(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Vector-based semantic search"""
        query_embedding = self.embeddings.embed_query(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        return [
            {
                'content': doc,
                'metadata': meta,
                'score': distance,
                'type': 'vector'
            }
            for doc, meta, distance in zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )
        ]
    
    async def _kg_retrieval(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Knowledge graph pattern matching"""
        with self.neo4j_driver.session() as session:
            result = session.run("""
                MATCH (c:Company)-[r]->(m:Metric)
                WHERE c.name CONTAINS $query OR m.name CONTAINS $query
                RETURN c.name as company, type(r) as relationship, m.name as metric, m.value as value
                LIMIT $top_k
            """, query=query, top_k=top_k)
            
            return [
                {
                    'content': f"{record['company']} {record['relationship']} {record['metric']}: {record['value']}",
                    'metadata': {'source': 'knowledge_graph', 'type': 'fact'},
                    'score': 0.9,  # KG results get high confidence
                    'type': 'knowledge_graph'
                }
                for record in result
            ]
    
    async def _add_to_knowledge_graph(self, document: Dict[str, Any]):
        """Extract entities and relationships for knowledge graph"""
        # Implementation for NER and relationship extraction
        # This would use an LLM to extract financial entities and relationships
        pass
    
    def _rerank_results(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Simple reranking based on type and score"""
        for result in results:
            # Boost KG results for factual queries
            if result['type'] == 'knowledge_graph' and any(word in query.lower() for word in ['what', 'when', 'who', 'how much']):
                result['score'] *= 1.2
        
        return sorted(results, key=lambda x: x['score'], reverse=True)