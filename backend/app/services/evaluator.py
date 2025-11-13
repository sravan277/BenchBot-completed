import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from app.services.rag_pipeline import RAGPipeline, EmbeddingModelManager

class BenchmarkEvaluator:
    """Evaluates RAG pipeline performance"""
    
    def __init__(self, corpus_dir: Path):
        self.corpus_dir = Path(corpus_dir)
        if not self.corpus_dir.exists():
            # Try relative path from project root (go up from backend/app/services to project root)
            project_root = Path(__file__).parent.parent.parent.parent
            self.corpus_dir = project_root / "data" / "corpse_data"
            if not self.corpus_dir.exists():
                raise FileNotFoundError(f"Corpus directory not found: {corpus_dir} or {self.corpus_dir}")
        print(f"Using corpus directory: {self.corpus_dir.absolute()}")
        self.embedding_manager = EmbeddingModelManager()
        self.progress_callback = None
    
    def set_progress_callback(self, callback):
        """Set callback function for progress updates: callback(step, progress, details)"""
        self.progress_callback = callback
    
    def _update_progress(self, step: str, progress: float, details: str = ""):
        """Internal method to update progress"""
        if self.progress_callback:
            self.progress_callback(step, progress, details)
    
    def load_ground_truth(self, benchmark_type: str, language: str = "en") -> List[Dict]:
        """Load ground truth data"""
        if benchmark_type == "single":
            gt_dir = self.corpus_dir.parent / "Ground_truth" / "single_JSON"
            if language == "en":
                gt_path = gt_dir / "JSON_data_english"
            elif language == "hi":
                gt_path = gt_dir / "JSON_data_hindi"
            elif language == "te":
                gt_path = gt_dir / "JSON_data_telugu"
            else:
                raise ValueError(f"Unsupported language: {language}")
            
            all_queries = []
            for json_file in sorted(gt_path.glob("*.json")):
                with open(json_file, "r", encoding="utf-8") as f:
                    queries = json.load(f)
                    all_queries.extend(queries)
            return all_queries
        
        elif benchmark_type == "multi_hop":
            gt_dir = self.corpus_dir.parent / "Ground_truth" / "multi_JSON"
            if language == "en":
                gt_path = gt_dir / "multi_english"
            elif language == "hi":
                gt_path = gt_dir / "multi_hindi"
            elif language == "te":
                gt_path = gt_dir / "multi_telugu"
            else:
                raise ValueError(f"Unsupported language: {language}")
            
            all_queries = []
            for json_file in sorted(gt_path.glob("*.json")):
                with open(json_file, "r", encoding="utf-8") as f:
                    queries = json.load(f)
                    all_queries.extend(queries)
            return all_queries
        
        elif benchmark_type == "multilingual":
            # Load all languages for single document queries
            all_queries = []
            for lang in ["en", "hi", "te"]:
                queries = self.load_ground_truth("single", lang)
                all_queries.extend(queries)
            return all_queries
        else:
            raise ValueError(f"Unsupported benchmark type: {benchmark_type}")
    
    def calculate_span_overlap(self, retrieved_span: Tuple[int, int], ground_truth_spans: List[List[int]]) -> float:
        """Calculate overlap between retrieved span and ground truth spans"""
        if not ground_truth_spans:
            return 0.0
        
        ret_start, ret_end = retrieved_span
        max_overlap = 0.0
        
        for gt_span in ground_truth_spans:
            gt_start, gt_end = gt_span[0], gt_span[1]
            
            # Calculate intersection
            overlap_start = max(ret_start, gt_start)
            overlap_end = min(ret_end, gt_end)
            
            if overlap_start < overlap_end:
                overlap_length = overlap_end - overlap_start
                ret_length = ret_end - ret_start
                gt_length = gt_end - gt_start
                
                # IoU (Intersection over Union)
                union_length = ret_length + gt_length - overlap_length
                if union_length > 0:
                    iou = overlap_length / union_length
                    max_overlap = max(max_overlap, iou)
        
        return max_overlap
    
    def calculate_similarity_score(self, retrieved_text: str, ground_truth_snippets: List[str], model_name: str) -> float:
        """Calculate semantic similarity between retrieved text and ground truth"""
        if not ground_truth_snippets:
            return 0.0
        
        try:
            texts = [retrieved_text] + ground_truth_snippets
            embeddings = self.embedding_manager.encode(model_name, texts)
            
            query_emb = embeddings[0:1]
            gt_embs = embeddings[1:]
            
            similarities = cosine_similarity(query_emb, gt_embs)[0]
            return float(np.max(similarities))
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0
    
    def evaluate_query(
        self,
        query_data: Dict,
        retrieved_chunks: List[Dict],
        pipeline: RAGPipeline,
        benchmark_type: str
    ) -> Dict[str, Any]:
        """Evaluate a single query"""
        labels = query_data.get("labels", [])
        
        # Extract ground truth spans
        gt_spans_by_file = {}
        gt_snippets = []
        
        for label in labels:
            file_name = label["file_name"]
            spans = label.get("spans", [])
            snippet = label.get("snippet", "")
            
            if file_name not in gt_spans_by_file:
                gt_spans_by_file[file_name] = []
            gt_spans_by_file[file_name].extend(spans)
            gt_snippets.append(snippet)
        
        # Calculate metrics
        true_positives = 0
        false_positives = 0
        total_gt_spans = sum(len(spans) for spans in gt_spans_by_file.values())
        
        similarity_scores = []
        
        for chunk in retrieved_chunks:
            file_name = chunk["file_name"]
            start_pos = chunk["start_pos"]
            end_pos = chunk["end_pos"]
            chunk_text = chunk.get("chunk_text", "")
            
            # Check if this chunk overlaps with ground truth
            if file_name in gt_spans_by_file:
                overlap = self.calculate_span_overlap(
                    (start_pos, end_pos),
                    gt_spans_by_file[file_name]
                )
                
                if overlap > 0.3:  # Threshold for considering it a match
                    true_positives += 1
                else:
                    false_positives += 1
                
                # Calculate similarity
                sim_score = self.calculate_similarity_score(
                    chunk_text,
                    gt_snippets,
                    pipeline.embedding_model
                )
                similarity_scores.append(sim_score)
            else:
                false_positives += 1
        
        # Calculate precision, recall, F1
        retrieved_count = len(retrieved_chunks)
        precision = true_positives / retrieved_count if retrieved_count > 0 else 0.0
        recall = true_positives / total_gt_spans if total_gt_spans > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        avg_similarity = np.mean(similarity_scores) if similarity_scores else 0.0
        
        return {
            "query_id": query_data.get("query_id", ""),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1_score),
            "similarity_score": float(avg_similarity),
            "true_positives": true_positives,
            "false_positives": false_positives,
            "total_gt_spans": total_gt_spans
        }
    
    def evaluate_pipeline(
        self,
        pipeline: RAGPipeline,
        benchmark_type: str,
        languages: List[str] = None,
        job_id: str = None,
        base_progress: float = 0.0,
        progress_range: float = 100.0
    ) -> Dict[str, Any]:
        """Evaluate entire pipeline"""
        if languages is None:
            languages = ["en"]
        
        all_results = []
        total_latency = 0.0
        query_count = 0
        
        # Load and index corpus for all languages at once for multilingual
        progress_start = base_progress
        progress_end = base_progress + progress_range * 0.4  # 40% for loading/indexing
        
        if benchmark_type == "multilingual":
            self._update_progress(f"Loading corpus ({benchmark_type})", progress_start, f"Loading documents for {', '.join(languages)}...")
            all_documents = {}
            for idx, language in enumerate(languages):
                lang_progress = progress_start + (progress_end - progress_start) * (idx + 1) / len(languages)
                self._update_progress(f"Loading {language} corpus", lang_progress, f"Loading {language} documents...")
                lang_docs = pipeline.load_corpus(self.corpus_dir, language)
                all_documents.update(lang_docs)
            self._update_progress("Chunking documents", progress_start + (progress_end - progress_start) * 0.7, "Chunking documents...")
            def chunk_progress_callback(step, prog, det):
                # prog is 0.0-1.0, convert to absolute progress
                abs_progress = progress_start + (progress_end - progress_start) * (0.7 + prog * 0.3)
                self._update_progress(step, abs_progress, det)
            pipeline.chunk_and_index(all_documents, progress_callback=chunk_progress_callback)
            self._update_progress("Loading queries", progress_end, "Loading ground truth queries...")
            queries = self.load_ground_truth(benchmark_type, "en")  # Language doesn't matter for multilingual
        else:
            # For single and multi_hop, use first language
            language = languages[0] if languages else "en"
            self._update_progress(f"Loading corpus ({benchmark_type})", progress_start, f"Loading {language} documents...")
            print(f"Loading corpus for language: {language}")
            documents = pipeline.load_corpus(self.corpus_dir, language)
            self._update_progress("Chunking documents", progress_start + (progress_end - progress_start) * 0.3, "Chunking documents...")
            def chunk_progress_callback(step, prog, det):
                # prog is 0.0-1.0, convert to absolute progress
                abs_progress = progress_start + (progress_end - progress_start) * (0.3 + prog * 0.4)
                self._update_progress(step, abs_progress, det)
            pipeline.chunk_and_index(documents, progress_callback=chunk_progress_callback)
            self._update_progress("Loading queries", progress_end, "Loading ground truth queries...")
            queries = self.load_ground_truth(benchmark_type, language)
        
        # Evaluate each query
        query_progress_start = progress_end
        query_progress_end = base_progress + progress_range * 0.95
        total_queries = len(queries)
        
        for idx, query_data in enumerate(queries):
            query = query_data.get("query", "")
            query_progress = query_progress_start + (query_progress_end - query_progress_start) * (idx + 1) / total_queries
            self._update_progress(
                f"Evaluating queries ({benchmark_type})",
                query_progress,
                f"Processing query {idx + 1}/{total_queries}..."
            )
            
            retrieved_chunks, latency = pipeline.retrieve(query)
            
            result = self.evaluate_query(query_data, retrieved_chunks, pipeline, benchmark_type)
            result["latency"] = latency
            
            # Set language for multilingual
            if benchmark_type == "multilingual":
                result["language"] = query_data.get("language", "en")
            else:
                result["language"] = language
            
            all_results.append(result)
            
            total_latency += latency
            query_count += 1
        
        self._update_progress("Calculating metrics", base_progress + progress_range * 0.95, "Calculating final metrics...")
        
        # Calculate aggregate metrics
        avg_precision = np.mean([r["precision"] for r in all_results])
        avg_recall = np.mean([r["recall"] for r in all_results])
        avg_f1 = np.mean([r["f1_score"] for r in all_results])
        avg_similarity = np.mean([r["similarity_score"] for r in all_results])
        avg_latency = total_latency / query_count if query_count > 0 else 0.0
        
        # Per-language metrics for multilingual
        language_metrics = {}
        if benchmark_type == "multilingual":
            for lang in languages:
                lang_results = [r for r in all_results if r["language"] == lang]
                if lang_results:
                    language_metrics[lang] = {
                        "precision": float(np.mean([r["precision"] for r in lang_results])),
                        "recall": float(np.mean([r["recall"] for r in lang_results])),
                        "f1_score": float(np.mean([r["f1_score"] for r in lang_results])),
                        "similarity_score": float(np.mean([r["similarity_score"] for r in lang_results]))
                    }
        
        return {
            "overall_metrics": {
                "precision": float(avg_precision),
                "recall": float(avg_recall),
                "f1_score": float(avg_f1),
                "similarity_score": float(avg_similarity),
                "latency": float(avg_latency),
                "query_count": query_count
            },
            "per_query_results": all_results,
            "language_metrics": language_metrics if language_metrics else None
        }

