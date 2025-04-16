import logging
import uuid
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct, VectorParams, Distance,
    SparseVectorParams, SparseVector, Prefetch, FusionQuery, Fusion
)
from fastembed import SparseTextEmbedding
from typing import List, Dict, Any

# Get logger
logger = logging.getLogger('search_engine')

class HybridSearchEngine:
    """Enhanced search engine with hybrid search capabilities"""
    
    def __init__(self, config):
        logger.info("Initializing HybridSearchEngine")
        # Initialize clients and models
        self.openai_client = OpenAI(api_key=config.openai_api_key)
        self.qdrant_client = QdrantClient(url=config.qdrant_url, api_key=config.qdrant_api_key)
        self.collection_name = config.collection_name
        self.sparse_model = SparseTextEmbedding(model_name="prithvida/Splade_PP_en_v1")
        self.llm_model = config.llm_model
        logger.info("Clients and models initialized")
        self.setup_collection()
    
    def setup_collection(self):
        logger.info(f"Setting up collection: {self.collection_name}")
        # Create collection if it doesn't exist
        if not self.qdrant_client.collection_exists(collection_name=self.collection_name):
            logger.info("Collection does not exist, creating new collection")
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config={"dense": VectorParams(size=512, distance=Distance.COSINE)},
                sparse_vectors_config={"sparse": SparseVectorParams()}
            )
            logger.info("Collection created successfully")
        else:
            logger.info("Collection already exists")
    
    def get_dense_embedding(self, text: str) -> List[float]:
        logger.debug("Generating dense embedding")
        # Generate dense embedding
        text = text.replace("\n", " ")
        try:
            response = self.openai_client.embeddings.create(
                input=[text], 
                model="text-embedding-3-small", 
                dimensions=512
            )
            logger.debug("Dense embedding generated successfully")
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating dense embedding: {str(e)}")
            raise
    
    def get_sparse_embedding(self, text: str) -> SparseVector:
        logger.debug("Generating sparse embedding")
        # Generate sparse embedding
        try:
            embedding = next(self.sparse_model.embed([text]))
            logger.debug("Sparse embedding generated successfully")
            return SparseVector(indices=embedding.indices.tolist(), values=embedding.values.tolist())
        except Exception as e:
            logger.error(f"Error generating sparse embedding: {str(e)}")
            raise
    
    def index_document(self, document: Dict[str, Any]) -> int:
        logger.info(f"Indexing document with ID: {document.get('id', 'new')}")
        try:
            # Process and index a single document
            combined_text = f"Question: {document['question']} Answer: {document['answer']}"
            logger.debug("Generating embeddings for document")
            dense_vec = self.get_dense_embedding(combined_text)
            sparse_vec = self.get_sparse_embedding(combined_text)
            
            metadata = {
                "question": document['question'],
                "answer": document['answer'],
                "summary": document.get("summary", ""),
                "answer_type": document.get("answer_type", ""),
                "date": document.get("date", "")
            }
            
            # Generate a UUID for the point ID if not provided
            point_id = document.get('id', str(uuid.uuid4()))
            
            point = PointStruct(
                id=point_id,
                vector={"dense": dense_vec, "sparse": sparse_vec.dict()},
                payload=metadata
            )
            
            logger.debug("Upserting document to Qdrant")
            self.qdrant_client.upsert(collection_name=self.collection_name, points=[point])
            logger.info(f"Document indexed successfully with ID: {point.id}")
            return point.id
        except Exception as e:
            logger.error(f"Error indexing document: {str(e)}")
            raise
    
    def bulk_index_documents(self, documents: List[Dict[str, Any]]) -> int:
        logger.info(f"Bulk indexing {len(documents)} documents")
        try:
            # Process and index multiple documents
            points = []
            for i, document in enumerate(documents, 1):
                logger.debug(f"Processing document {i}/{len(documents)}")
                combined_text = f"Question: {document['question']} Answer: {document['answer']}"
                dense_vec = self.get_dense_embedding(combined_text)
                sparse_vec = self.get_sparse_embedding(combined_text)
                
                metadata = {
                    "question": document['question'],
                    "answer": document['answer'],
                    "summary": document.get("summary", ""),
                    "answer_type": document.get("answer_type", ""),
                    "date": document.get("date", "")
                }
                
                # Generate a UUID for the point ID if not provided
                point_id = document.get('id', str(uuid.uuid4()))
                
                point = PointStruct(
                    id=point_id,
                    vector={"dense": dense_vec, "sparse": sparse_vec.dict()},
                    payload=metadata
                )
                points.append(point)
            
            logger.debug("Upserting all documents to Qdrant")
            self.qdrant_client.upsert(collection_name=self.collection_name, points=points)
            logger.info(f"Successfully indexed {len(points)} documents")
            return len(points)
        except Exception as e:
            logger.error(f"Error bulk indexing documents: {str(e)}")
            raise
    
    def hybrid_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        logger.info(f"Performing hybrid search for query: {query}")
        try:
            # Perform hybrid search
            logger.debug("Generating embeddings for query")
            dense_vec = self.get_dense_embedding(query)
            sparse_vec = self.get_sparse_embedding(query)
            
            logger.debug("Executing hybrid search in Qdrant")
            results = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    Prefetch(query=dense_vec, using="dense", limit=top_k),
                    Prefetch(query=sparse_vec.dict(), using="sparse", limit=top_k)
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                with_payload=True,
                limit=top_k
            )
            
            search_results = []
            for result in results.points:
                search_results.append({
                    "id": result.id,
                    "score": result.score,
                    "payload": result.payload
                })
            
            logger.info(f"Found {len(search_results)} results")
            return search_results
        except Exception as e:
            logger.error(f"Error performing hybrid search: {str(e)}")
            raise
    
    def generate_answer(self, query: str, search_results: List[Dict[str, Any]], top_k: int = 5) -> Dict[str, Any]:
        logger.info("Generating answer using LLM")
        try:
            # Generate answer using LLM
            if not search_results:
                logger.warning("No search results found")
                return {
                    "answer": "No relevant information found.",
                    "confidence": 0.0,
                    "confidence_breakdown": {
                        "relevance": 0.0,
                        "diversity": 0.0,
                        "agreement": 0.0,
                        "coverage": 0.0
                    }
                }
            
            # Format context from search results
            logger.debug("Formatting context from search results")
            context = ""
            for i, result in enumerate(search_results):
                context += f"Source {i+1}:\n"
                context += f"Question: {result['payload']['question']}\n"
                context += f"Answer: {result['payload']['answer']}\n"
                if result['payload'].get("summary"):
                    context += f"Summary: {result['payload']['summary']}\n"
                context += f"Relevance Score: {result['score']:.2f}\n\n"
            
            # Create a prompt for the LLM
            logger.debug("Creating prompt for LLM")
            prompt = f"""
                You are an RFP (Request for Proposal) answering assistant. 
                Use the provided context from a hybrid search to answer the user's question accurately.
                Only use information from the provided context. If the context doesn't contain enough 
                information to answer the question fully, acknowledge the limitations in your response.

                User Question: {query}

                Context from search results:
                {context}

                Instructions:
                1. Answer the question directly and precisely
                2. If multiple sources provide relevant information, synthesize them
                3. If information is incomplete, acknowledge it in your response
                4. Include any relevant dates, certifications, or specific details mentioned in the context
                5. Do not make up information that isn't explicitly stated in the context

                Your answer:
                """
            
            # Generate answer using OpenAI
            logger.debug(f"Calling OpenAI API with model: {self.llm_model}")
            response = self.openai_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are an RFP assistant that provides clear, accurate answers based on the retrieved information."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            # Calculate confidence score with multiple factors
            logger.debug("Calculating confidence score")
            
            # 1. Relevance Score (normalized)
            max_possible_score = 1.0  # Assuming cosine similarity scores are between 0 and 1
            top_relevance = search_results[0]["score"] / max_possible_score if search_results else 0
            
            # 2. Source Diversity
            source_diversity = min(len(search_results) / top_k, 1.0)
            
            # 3. Source Agreement (how similar the answers are)
            answers = [result['payload']['answer'] for result in search_results]
            answer_similarities = []
            for i in range(len(answers)):
                for j in range(i+1, len(answers)):
                    # Get embeddings for answer pairs
                    emb1 = self.get_dense_embedding(answers[i])
                    emb2 = self.get_dense_embedding(answers[j])
                    # Calculate cosine similarity
                    similarity = sum(a*b for a,b in zip(emb1, emb2)) / (
                        (sum(a*a for a in emb1) ** 0.5) * 
                        (sum(b*b for b in emb2) ** 0.5)
                    )
                    answer_similarities.append(similarity)
            source_agreement = sum(answer_similarities) / len(answer_similarities) if answer_similarities else 0
            
            # 4. Coverage Score (how well the answer covers the question)
            answer_embedding = self.get_dense_embedding(response.choices[0].message.content)
            question_embedding = self.get_dense_embedding(query)
            coverage = sum(a*b for a,b in zip(answer_embedding, question_embedding)) / (
                (sum(a*a for a in answer_embedding) ** 0.5) * 
                (sum(b*b for b in question_embedding) ** 0.5)
            )
            
            # Weighted confidence calculation
            weights = {
                'relevance': 0.4,      # Importance of top result relevance
                'diversity': 0.2,      # Importance of having multiple sources
                'agreement': 0.2,      # Importance of source consistency
                'coverage': 0.2        # Importance of answer completeness
            }
            
            confidence = (
                weights['relevance'] * top_relevance +
                weights['diversity'] * source_diversity +
                weights['agreement'] * source_agreement +
                weights['coverage'] * coverage
            )
            
            logger.info(f"Confidence breakdown - Relevance: {top_relevance:.2f}, "
                       f"Diversity: {source_diversity:.2f}, "
                       f"Agreement: {source_agreement:.2f}, "
                       f"Coverage: {coverage:.2f}, "
                       f"Final: {confidence:.2f}")
            
            return {
                "answer": response.choices[0].message.content,
                "confidence": confidence,
                "confidence_breakdown": {
                    "relevance": top_relevance,
                    "diversity": source_diversity,
                    "agreement": source_agreement,
                    "coverage": coverage
                }
            }
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            raise
    
    def search_and_answer(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        logger.info(f"Starting search and answer pipeline for query: {query}")
        try:
            # Complete search and answer pipeline
            search_results = self.hybrid_search(query, top_k)
            answer_data = self.generate_answer(query, search_results, top_k)
            
            logger.info("Search and answer pipeline completed")
            return {
                "query": query,
                "search_results": search_results,
                "answer": answer_data["answer"],
                "confidence": answer_data["confidence"],
                "confidence_breakdown": answer_data["confidence_breakdown"]
            }
        except Exception as e:
            logger.error(f"Error in search and answer pipeline: {str(e)}")
            raise 