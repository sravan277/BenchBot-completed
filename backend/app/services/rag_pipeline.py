import os
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import BertModel, BertTokenizerFast, AutoTokenizer, AutoModel
import chromadb
from chromadb.config import Settings
import faiss
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import re

class EmbeddingModelManager:
    """Manages embedding models with caching"""
    
    def __init__(self):
        self.models = {}
        self.model_cache_dir = Path("models_cache")
        self.model_cache_dir.mkdir(exist_ok=True)
        # Check for GPU availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
        else:
            print("Using CPU (GPU not available)")
    
    def get_model(self, model_name: str):
        """Get or load embedding model"""
        if model_name in self.models:
            return self.models[model_name]
        
        print(f"Loading model: {model_name}")
        
        if model_name == "setu4993/LEALLA-base":
            # Special handling for LEALLA
            tokenizer = BertTokenizerFast.from_pretrained(
                model_name,
                cache_dir=str(self.model_cache_dir / "lealla")
            )
            model = BertModel.from_pretrained(
                model_name,
                cache_dir=str(self.model_cache_dir / "lealla")
            )
            model = model.to(self.device)  # Move to GPU if available
            model = model.eval()
            self.models[model_name] = ("lealla", model, tokenizer)
        elif model_name == "intfloat/multilingual-e5-small":
            # Special handling for multilingual-e5-small
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=str(self.model_cache_dir / "multilingual-e5-small")
            )
            model = AutoModel.from_pretrained(
                model_name,
                cache_dir=str(self.model_cache_dir / "multilingual-e5-small")
            )
            model = model.to(self.device)  # Move to GPU if available
            model = model.eval()
            self.models[model_name] = ("multilingual_e5", model, tokenizer)
        else:
            # Sentence transformers
            model = SentenceTransformer(
                model_name,
                cache_folder=str(self.model_cache_dir),
                device=self.device  # Use GPU if available
            )
            self.models[model_name] = ("sentence_transformer", model)
        
        return self.models[model_name]
    
    def encode(self, model_name: str, texts: List[str]) -> np.ndarray:
        """Encode texts using the specified model"""
        if not texts:
            return np.array([])
        
        try:
            model_data = self.get_model(model_name)
            if not model_data or len(model_data) < 2:
                raise ValueError(f"Invalid model data for {model_name}")
            
            model_type = model_data[0]
            
            if model_type == "lealla":
                if len(model_data) < 3:
                    raise ValueError(f"LEALLA model data incomplete: {model_data}")
                _, model, tokenizer = model_data
                inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
                # Move inputs to same device as model
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = model(**inputs)
                # Use CLS token embedding
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # Move back to CPU for numpy
                # Normalize
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / (norms + 1e-8)  # Add small epsilon to avoid division by zero
            elif model_type == "multilingual_e5":
                if len(model_data) < 3:
                    raise ValueError(f"Multilingual E5 model data incomplete: {model_data}")
                _, model, tokenizer = model_data
                
                # Add "query: " prefix to all texts (for retrieval tasks)
                prefixed_texts = [f"query: {text}" if not text.startswith(("query: ", "passage: ")) else text for text in texts]
                
                # Tokenize
                batch_dict = tokenizer(
                    prefixed_texts, 
                    max_length=512, 
                    padding=True, 
                    truncation=True, 
                    return_tensors='pt'
                )
                # Move to device
                batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
                
                with torch.no_grad():
                    outputs = model(**batch_dict)
                
                # Average pooling
                attention_mask = batch_dict['attention_mask']
                last_hidden = outputs.last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
                embeddings_tensor = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
                
                # Normalize
                embeddings_tensor = F.normalize(embeddings_tensor, p=2, dim=1)
                
                # Move back to CPU for numpy
                embeddings = embeddings_tensor.cpu().numpy()
            else:
                if len(model_data) < 2:
                    raise ValueError(f"Sentence transformer model data incomplete: {model_data}")
                _, model = model_data
                embeddings = model.encode(texts, convert_to_numpy=True)
            
            return embeddings
        except Exception as e:
            print(f"Error encoding texts with {model_name}: {e}")
            import traceback
            traceback.print_exc()
            # Return zero embeddings as fallback
            return np.zeros((len(texts), 384))  # Default dimension

class ChunkingStrategy:
    """Handles text chunking"""
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int, chunk_overlap: int = 0) -> List[Tuple[str, int, int]]:
        """
        Chunk text with character-based chunking
        Returns: List of (chunk_text, start_pos, end_pos)
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            
            # Try to break at word boundary
            if end < len(text):
                last_space = chunk_text.rfind(' ')
                if last_space > chunk_size * 0.5:  # Only break if we're not too far from end
                    end = start + last_space + 1
                    chunk_text = text[start:end]
            
            chunks.append((chunk_text, start, end))
            
            start = end - chunk_overlap
            if start >= len(text):
                break
        
        return chunks

class VectorDatabase:
    """Manages vector database operations"""
    
    def __init__(self, db_type: str = "chroma", collection_name: str = "rag_benchmark"):
        self.db_type = db_type
        self.collection_name = collection_name
        
        if db_type == "chroma":
            # Use new Chroma client API
            self.client = chromadb.PersistentClient(path="./chroma_db")
            self.collection = self.client.get_or_create_collection(name=collection_name)
        elif db_type == "faiss":
            self.index = None
            self.metadata = []
        elif db_type == "qdrant":
            # Initialize Qdrant client
            from dotenv import load_dotenv
            load_dotenv()
            qdrant_url = os.getenv("QDRANT_URL", "https://cbeb0a1f-14a5-4300-abef-f421ac777a71.us-west-1-0.aws.cloud.qdrant.io")
            qdrant_key = os.getenv("QDRANT_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.9jLr2vjtTssWw9RZ0yA2C6KCqtCxpC9S7VidzK8EiFs")
            print(f"Connecting to Qdrant at: {qdrant_url}")
            self.client = QdrantClient(
                url=qdrant_url,
                api_key=qdrant_key
            )
            self.collection_name = collection_name
            # Collection will be created when adding documents
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
    
    def add_documents(self, embeddings: np.ndarray, documents: List[str], metadatas: List[Dict]):
        """Add documents to vector database"""
        if self.db_type == "chroma":
            ids = [f"doc_{i}" for i in range(len(documents))]
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
        elif self.db_type == "faiss":
            dimension = embeddings.shape[1]
            if self.index is None:
                self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings.astype('float32'))
            # Store metadata with index information
            for i, meta in enumerate(metadatas):
                meta_with_idx = meta.copy()
                meta_with_idx["_index"] = len(self.metadata) + i
                self.metadata.append(meta_with_idx)
        elif self.db_type == "qdrant":
            dimension = embeddings.shape[1]
            # Create collection if it doesn't exist
            try:
                self.client.get_collection(self.collection_name)
            except:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=dimension,
                        distance=Distance.COSINE
                    )
                )
            
            # Prepare points for Qdrant
            points = []
            for i, (embedding, doc, metadata) in enumerate(zip(embeddings, documents, metadatas)):
                point_id = str(uuid.uuid4())
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=embedding.tolist(),
                        payload={
                            "document": doc,
                            "metadata": metadata
                        }
                    )
                )
            
            # Upload points in batches
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Search for similar documents"""
        if self.db_type == "chroma":
            try:
                results = self.collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=top_k
                )
                # Handle case where results might be empty
                if not results or "documents" not in results or not results["documents"]:
                    return []
                
                documents = results["documents"][0] if results["documents"] else []
                if not documents:
                    return []
                
                # Safely get metadatas and distances
                metadatas = results.get("metadatas", [])
                metadata_list = metadatas[0] if metadatas and len(metadatas) > 0 else [{}] * len(documents)
                
                distances = results.get("distances", [])
                distance_list = distances[0] if distances and len(distances) > 0 else [1.0] * len(documents)
                
                # Ensure all lists have the same length
                min_len = min(len(documents), len(metadata_list), len(distance_list))
                if min_len == 0:
                    return []
                
                return [
                    {
                        "document": doc,
                        "metadata": meta if meta else {},
                        "distance": dist
                    }
                    for doc, meta, dist in zip(
                        documents[:min_len],
                        metadata_list[:min_len],
                        distance_list[:min_len]
                    )
                ]
            except Exception as e:
                print(f"Error in ChromaDB search: {e}")
                import traceback
                traceback.print_exc()
                return []
        elif self.db_type == "faiss":
            if self.index is None or len(self.metadata) == 0:
                return []
            distances, indices = self.index.search(query_embedding.astype('float32').reshape(1, -1), top_k)
            results = []
            for idx, dist in zip(indices[0], distances[0]):
                if 0 <= idx < len(self.metadata):
                    meta = self.metadata[idx].copy()
                    meta.pop("_index", None)  # Remove internal index
                    results.append({
                        "metadata": meta,
                        "distance": float(dist),
                        "index": idx  # Store index for retrieving chunk text
                    })
            return results
        elif self.db_type == "qdrant":
            try:
                # Search in Qdrant
                search_results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding.tolist(),
                    limit=top_k
                )
                
                results = []
                for hit in search_results:
                    payload = hit.payload
                    results.append({
                        "document": payload.get("document", ""),
                        "metadata": payload.get("metadata", {}),
                        "distance": 1.0 - hit.score  # Convert similarity to distance
                    })
                return results
            except Exception as e:
                print(f"Error in Qdrant search: {e}")
                import traceback
                traceback.print_exc()
                return []

class RAGPipeline:
    """Main RAG pipeline class"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.embedding_manager = EmbeddingModelManager()
        self.embedding_model = config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        self.chunk_size = config.get("chunk_size", 500)
        self.chunk_overlap = config.get("chunk_overlap", 50)
        self.top_k = config.get("top_k", 5)
        self.vector_db_type = config.get("vector_db", "chroma")
        self.reranking_strategy = config.get("reranking_strategy", "none")
        self.reranker = None
        self.bm25_index = None
        
        self.vector_db = VectorDatabase(
            db_type=self.vector_db_type,
            collection_name=f"rag_{int(time.time())}"
        )
        self.chunks = []
        self.chunk_metadata = []
    
    def load_corpus(self, corpus_dir: Path, language: str = "en"):
        """Load corpus files based on language"""
        corpus_dir = Path(corpus_dir)
        if not corpus_dir.exists():
            raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")
        
        if language == "en":
            corpus_path = corpus_dir / "MA_Topics_english"
        elif language == "hi":
            corpus_path = corpus_dir / "MA_Topics_hindi"
        elif language == "te":
            corpus_path = corpus_dir / "MA_topics_telugu"
        else:
            raise ValueError(f"Unsupported language: {language}")
        
        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus path not found: {corpus_path}")
        
        documents = {}
        txt_files = list(corpus_path.glob("*.txt"))
        print(f"Found {len(txt_files)} text files in {corpus_path}")
        
        if len(txt_files) == 0:
            raise ValueError(f"No .txt files found in {corpus_path}")
        
        for file_path in txt_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    if content.strip():  # Only add non-empty files
                        documents[file_path.name] = content
                        print(f"Loaded {file_path.name} ({len(content)} characters)")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        print(f"Loaded {len(documents)} documents")
        return documents
    
    def chunk_and_index(self, documents: Dict[str, str], progress_callback=None):
        """Chunk documents and create embeddings"""
        self.chunks = []
        self.chunk_metadata = []
        
        if not documents:
            raise ValueError("No documents provided for chunking")
        
        total_docs = len(documents)
        print(f"Chunking {total_docs} documents...")
        
        if progress_callback:
            progress_callback("Chunking documents", 0.0, f"Processing {total_docs} documents...")
        
        valid_docs = [d for d in documents.items() if d[1] and d[1].strip()]
        total_valid_docs = len(valid_docs)
        
        for doc_idx, (filename, text) in enumerate(valid_docs):
            if progress_callback and total_valid_docs > 0:
                chunk_progress = (doc_idx / total_valid_docs) * 0.3
                progress_callback("Chunking documents", chunk_progress, f"Chunking {filename} ({doc_idx + 1}/{total_valid_docs})...")
            
            chunks = ChunkingStrategy.chunk_text(text, self.chunk_size, self.chunk_overlap)
            print(f"Created {len(chunks)} chunks from {filename}")
            for chunk_text, start_pos, end_pos in chunks:
                self.chunks.append(chunk_text)
                self.chunk_metadata.append({
                    "file_name": filename,
                    "start_pos": start_pos,
                    "end_pos": end_pos
                })
        
        if len(self.chunks) == 0:
            raise ValueError("No chunks created from documents. Check if documents contain text.")
        
        # Generate embeddings in batches for better performance
        print(f"Generating embeddings for {len(self.chunks)} chunks...")
        if progress_callback:
            progress_callback("Generating embeddings", 0.3, f"Generating embeddings for {len(self.chunks)} chunks...")
        
        try:
            batch_size = 32  # Process in batches to avoid memory issues
            all_embeddings = []
            total_batches = (len(self.chunks) + batch_size - 1) // batch_size
            
            for i in range(0, len(self.chunks), batch_size):
                batch = self.chunks[i:i + batch_size]
                batch_num = i // batch_size + 1
                print(f"Processing batch {batch_num}/{total_batches}...")
                
                if progress_callback:
                    embed_progress = 0.3 + ((batch_num - 1) / total_batches) * 0.5
                    progress_callback("Generating embeddings", embed_progress, f"Processing batch {batch_num}/{total_batches}...")
                
                batch_embeddings = self.embedding_manager.encode(self.embedding_model, batch)
                all_embeddings.append(batch_embeddings)
                
                # Update progress after each batch
                if progress_callback:
                    embed_progress = 0.3 + (batch_num / total_batches) * 0.5
                    progress_callback("Generating embeddings", embed_progress, f"Completed batch {batch_num}/{total_batches}...")
            
            embeddings = np.vstack(all_embeddings)
            if embeddings.shape[0] != len(self.chunks):
                raise ValueError(f"Embedding shape mismatch: {embeddings.shape[0]} != {len(self.chunks)}")
            print(f"Successfully generated {embeddings.shape[0]} embeddings")
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Add to vector database
        print(f"Adding {len(self.chunks)} chunks to vector database...")
        if progress_callback:
            progress_callback("Uploading to database", 0.8, f"Uploading {len(self.chunks)} chunks to vector database...")
        
        self.vector_db.add_documents(embeddings, self.chunks, self.chunk_metadata)
        
        if progress_callback:
            progress_callback("Indexing complete", 1.0, f"Successfully indexed {len(self.chunks)} chunks")
        
        print(f"Successfully indexed {len(self.chunks)} chunks")
        return len(self.chunks)
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25"""
        # Convert to lowercase and split on whitespace and punctuation
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def _build_bm25_index(self):
        """Build BM25 index from chunks"""
        if self.bm25_index is not None:
            return
        
        tokenized_chunks = [self._tokenize(chunk) for chunk in self.chunks]
        self.bm25_index = BM25Okapi(tokenized_chunks)
        print("BM25 index built")
    
    def _rerank_cross_encoder(self, query: str, chunks: List[str], scores: List[float]) -> List[Tuple[int, float]]:
        """Rerank using cross-encoder model"""
        if self.reranker is None:
            # Load cross-encoder model (lightweight and fast)
            model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
            try:
                self.reranker = CrossEncoder(model_name, max_length=512)
                print(f"Loaded cross-encoder model: {model_name}")
            except Exception as e:
                print(f"Error loading cross-encoder: {e}, falling back to no reranking")
                return list(enumerate(scores))
        
        # Create query-document pairs
        pairs = [[query, chunk] for chunk in chunks]
        
        # Get reranking scores
        try:
            rerank_scores = self.reranker.predict(pairs)
            # Combine with original scores (weighted average)
            combined_scores = [
                0.7 * float(rerank_score) + 0.3 * original_score
                for rerank_score, original_score in zip(rerank_scores, scores)
            ]
            # Return sorted indices with scores
            indexed_scores = list(enumerate(combined_scores))
            indexed_scores.sort(key=lambda x: x[1], reverse=True)
            return indexed_scores
        except Exception as e:
            print(f"Error in cross-encoder reranking: {e}")
            return list(enumerate(scores))
    
    def _rerank_bm25(self, query: str, chunks: List[str], scores: List[float]) -> List[Tuple[int, float]]:
        """Rerank using BM25"""
        if self.bm25_index is None:
            self._build_bm25_index()
        
        try:
            query_tokens = self._tokenize(query)
            bm25_scores = self.bm25_index.get_scores(query_tokens)
            
            # Combine BM25 scores with original embedding scores
            combined_scores = [
                0.6 * float(bm25_score) + 0.4 * original_score
                for bm25_score, original_score in zip(bm25_scores, scores)
            ]
            
            # Return sorted indices with scores
            indexed_scores = list(enumerate(combined_scores))
            indexed_scores.sort(key=lambda x: x[1], reverse=True)
            return indexed_scores
        except Exception as e:
            print(f"Error in BM25 reranking: {e}")
            return list(enumerate(scores))
    
    def _rerank_rrf(self, query: str, chunks: List[str], scores: List[float]) -> List[Tuple[int, float]]:
        """Rerank using Reciprocal Rank Fusion"""
        # RRF combines multiple ranking signals
        # For now, we'll use a simple combination of embedding scores
        # In a full implementation, you'd combine multiple retrieval methods
        
        # Normalize scores to ranks
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        # RRF formula: score = sum(1 / (k + rank))
        k = 60  # RRF constant
        rrf_scores = {}
        for rank, (idx, score) in enumerate(indexed_scores, 1):
            rrf_scores[idx] = 1.0 / (k + rank)
        
        # Combine with original scores
        combined_scores = [
            (idx, 0.5 * rrf_scores.get(idx, 0) + 0.5 * score)
            for idx, score in enumerate(scores)
        ]
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        return combined_scores
    
    def retrieve(self, query: str) -> Tuple[List[Dict], float]:
        """Retrieve relevant chunks for a query"""
        start_time = time.time()
        
        try:
            # Encode query
            query_embeddings = self.embedding_manager.encode(self.embedding_model, [query])
            if len(query_embeddings) == 0 or query_embeddings.shape[0] == 0:
                return [], time.time() - start_time
            query_embedding = query_embeddings[0]
            
            # Initial retrieval - get more candidates for reranking
            initial_top_k = self.top_k * 3 if self.reranking_strategy != "none" else self.top_k
            results = self.vector_db.search(query_embedding, top_k=initial_top_k)
        except Exception as e:
            print(f"Error in retrieve: {e}")
            import traceback
            traceback.print_exc()
            return [], time.time() - start_time
        
        # Format initial results
        retrieved_chunks = []
        chunk_texts = []
        scores = []
        
        for result in results:
            metadata = result.get("metadata", {})
            # Get chunk text from document if available, otherwise from chunks list
            chunk_text = result.get("document", "")
            if not chunk_text:
                # Try to get from chunks list using index
                idx = result.get("index", -1)
                if 0 <= idx < len(self.chunks):
                    chunk_text = self.chunks[idx]
            
            score = 1 - result.get("distance", 1.0)  # Convert distance to similarity
            
            retrieved_chunks.append({
                "file_name": metadata.get("file_name", ""),
                "start_pos": metadata.get("start_pos", 0),
                "end_pos": metadata.get("end_pos", 0),
                "chunk_text": chunk_text,
                "score": score
            })
            chunk_texts.append(chunk_text)
            scores.append(score)
        
        # Apply reranking if specified
        if self.reranking_strategy != "none" and len(retrieved_chunks) > 0:
            try:
                if self.reranking_strategy == "cross_encoder":
                    reranked_indices = self._rerank_cross_encoder(query, chunk_texts, scores)
                elif self.reranking_strategy == "bm25":
                    reranked_indices = self._rerank_bm25(query, chunk_texts, scores)
                elif self.reranking_strategy == "rrf":
                    reranked_indices = self._rerank_rrf(query, chunk_texts, scores)
                else:
                    reranked_indices = list(enumerate(scores))
                
                # Reorder chunks based on reranking
                reranked_chunks = [retrieved_chunks[idx] for idx, _ in reranked_indices[:self.top_k]]
                # Update scores
                for i, (_, score) in enumerate(reranked_indices[:self.top_k]):
                    reranked_chunks[i]["score"] = score
                
                retrieved_chunks = reranked_chunks
            except Exception as e:
                print(f"Error in reranking: {e}")
                # Fall back to original results
                retrieved_chunks = retrieved_chunks[:self.top_k]
        else:
            # No reranking, just take top_k
            retrieved_chunks = retrieved_chunks[:self.top_k]
        
        latency = time.time() - start_time
        return retrieved_chunks, latency

