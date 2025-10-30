from typing import List, Dict, Any, Optional
import os
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from config import Config

class MultiModalVectorStore:
    def __init__(self):
        self._setup_embeddings()
        self._setup_query_analyzer()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        
        # Initialize separate collections for different content types
        self.text_store = Chroma(
            collection_name=f"{Config.COLLECTION_NAME}_text",
            embedding_function=self.embeddings,
            persist_directory=f"{Config.PERSIST_DIRECTORY}/text"
        )
        
        self.table_store = Chroma(
            collection_name=f"{Config.COLLECTION_NAME}_tables",
            embedding_function=self.embeddings,
            persist_directory=f"{Config.PERSIST_DIRECTORY}/tables"
        )
        
        self.visual_store = Chroma(
            collection_name=f"{Config.COLLECTION_NAME}_visuals",
            embedding_function=self.embeddings,
            persist_directory=f"{Config.PERSIST_DIRECTORY}/visuals"
        )
    
    def _setup_embeddings(self):
        """Setup embeddings with service account authentication"""
        try:
            # Set environment variable for Google Application Credentials
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = Config.GOOGLE_CREDENTIALS_PATH
            
            self.embeddings = VertexAIEmbeddings(
                model_name="google-text-embedding-004"
            )
        except Exception as e:
            raise Exception(f"Failed to setup embeddings: {e}")
    
    def _setup_query_analyzer(self):
        """Setup LLM for query analysis"""
        try:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = Config.GOOGLE_CREDENTIALS_PATH
            self.query_analyzer_llm = ChatVertexAI(
                model_name="gemini-pro",
                temperature=0.1
            )
        except Exception as e:
            raise Exception(f"Failed to setup query analyzer LLM: {e}")
    
    def add_documents(self, processed_content: Dict[str, List[Dict[str, Any]]]):
        """Add processed documents to respective vector stores"""
        
        # Process text content
        text_docs = []
        for item in processed_content["text"]:
            chunks = self.text_splitter.split_text(item["content"])
            for chunk in chunks:
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": item["source"],
                        "page": item["page"],
                        "type": item["type"],
                        "image_path": item.get("image_path", "")
                    }
                )
                text_docs.append(doc)
        
        if text_docs:
            self.text_store.add_documents(text_docs)
        
        # Process table content
        table_docs = []
        for item in processed_content["tables"]:
            doc = Document(
                page_content=item["content"],
                metadata={
                    "source": item["source"],
                    "page": item["page"],
                    "type": item["type"],
                    "image_path": item.get("image_path", "")
                }
            )
            table_docs.append(doc)
        
        if table_docs:
            self.table_store.add_documents(table_docs)
        
        # Process visual content
        visual_docs = []
        for item in processed_content["visuals"]:
            doc = Document(
                page_content=item["content"],
                metadata={
                    "source": item["source"],
                    "page": item["page"],
                    "type": item["type"],
                    "image_path": item.get("image_path", "")
                }
            )
            visual_docs.append(doc)
        
        if visual_docs:
            self.visual_store.add_documents(visual_docs)
    
    def search_relevant_content(self, query: str, k: int = 3) -> Dict[str, List[Document]]:
        """Search across all content types and return relevant documents"""
        results = {
            "text": self.text_store.similarity_search(query, k=k),
            "tables": self.table_store.similarity_search(query, k=k),
            "visuals": self.visual_store.similarity_search(query, k=k)
        }
        return results
    
    def _analyze_query_intent(self, query: str) -> Dict[str, int]:
        """Use LLM to analyze query intent and determine retrieval strategy"""
        analysis_prompt = f"""
        Analyze the following user query and determine what type of information they are looking for.
        
        Query: "{query}"
        
        Based on the query, provide retrieval weights (1-5) for each content type:
        - text: For general textual information, paragraphs, explanations
        - tables: For structured data, numbers, statistics, comparisons
        - visuals: For charts, graphs, diagrams, visual elements
        
        Consider:
        1. What is the primary intent of the question?
        2. What type of content would best answer this question?
        3. How much emphasis should be placed on each content type?
        
        Respond in this exact format:
        text: [weight 1-5]
        tables: [weight 1-5]
        visuals: [weight 1-5]
        reasoning: [brief explanation]
        
        Example:
        text: 3
        tables: 5
        visuals: 2
        reasoning: Query asks for specific data comparison which is best found in tables
        """
        
        try:
            response = self.query_analyzer_llm.invoke([HumanMessage(content=analysis_prompt)])
            return self._parse_analysis_response(response.content)
        except Exception as e:
            print(f"Error in query analysis: {e}")
            # Fallback to balanced retrieval
            return {"text": 3, "tables": 2, "visuals": 2}
    
    def _parse_analysis_response(self, response: str) -> Dict[str, int]:
        """Parse LLM response to extract retrieval weights"""
        weights = {"text": 3, "tables": 2, "visuals": 2}  # Default values
        
        try:
            lines = response.strip().split('\n')
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    if key in weights:
                        try:
                            weight = int(value.strip())
                            weights[key] = max(1, min(5, weight))  # Clamp between 1-5
                        except ValueError:
                            continue
        except Exception as e:
            print(f"Error parsing analysis response: {e}")
        
        return weights
    
    def get_contextual_documents(self, query: str) -> List[Document]:
        """Get contextually relevant documents using LLM-driven retrieval strategy"""
        # Use LLM to analyze query intent
        weights = self._analyze_query_intent(query)
        
        # Convert weights to retrieval counts (scale weights to reasonable k values)
        max_total_docs = 9  # Maximum total documents to retrieve
        total_weight = sum(weights.values())
        
        text_k = max(1, int((weights["text"] / total_weight) * max_total_docs))
        table_k = max(1, int((weights["tables"] / total_weight) * max_total_docs))
        visual_k = max(1, int((weights["visuals"] / total_weight) * max_total_docs))
        
        print(f"LLM-driven retrieval strategy - Text: {text_k}, Tables: {table_k}, Visuals: {visual_k}")
        
        results = {
            "text": self.text_store.similarity_search(query, k=text_k),
            "tables": self.table_store.similarity_search(query, k=table_k),
            "visuals": self.visual_store.similarity_search(query, k=visual_k)
        }
        
        # Combine and return all relevant documents
        all_docs = []
        for doc_type, docs in results.items():
            all_docs.extend(docs)
        
        return all_docs
