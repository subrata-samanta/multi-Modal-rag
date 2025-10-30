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
        
        # Use smaller chunk sizes to avoid token limit issues
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Reduced from default
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
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
    
    def _chunk_large_content(self, content: str, max_chunk_size: int = 1000) -> List[str]:
        """Chunk large content to avoid token limits"""
        chunks = self.text_splitter.split_text(content)
        
        # Further split if chunks are still too large
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > max_chunk_size * 4:  # Rough token estimate (4 chars per token)
                # Split by sentences if chunk is too large
                sentences = chunk.split('. ')
                current_chunk = ""
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) < max_chunk_size * 4:
                        current_chunk += sentence + ". "
                    else:
                        if current_chunk:
                            final_chunks.append(current_chunk.strip())
                        current_chunk = sentence + ". "
                if current_chunk:
                    final_chunks.append(current_chunk.strip())
            else:
                final_chunks.append(chunk)
        
        return final_chunks
    
    def add_documents(self, processed_content: Dict[str, List[Dict[str, Any]]]):
        """Add processed documents to respective vector stores with batching"""
        from tqdm import tqdm
        
        # Process text content
        text_docs = []
        print("Processing text content...")
        for item in tqdm(processed_content["text"], desc="Chunking text", unit="page"):
            try:
                chunks = self._chunk_large_content(item["content"])
                for chunk_idx, chunk in enumerate(chunks):
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "source": item["source"],
                            "page": item["page"],
                            "type": item["type"],
                            "chunk_index": chunk_idx,
                            "image_path": item.get("image_path", "")
                        }
                    )
                    text_docs.append(doc)
            except Exception as e:
                print(f"Error chunking text from page {item['page']}: {e}")
        
        # Add text documents in batches
        if text_docs:
            batch_size = 50
            print(f"Adding {len(text_docs)} text chunks to vector store...")
            for i in tqdm(range(0, len(text_docs), batch_size), desc="Adding text batches"):
                batch = text_docs[i:i+batch_size]
                try:
                    self.text_store.add_documents(batch)
                except Exception as e:
                    print(f"Error adding text batch {i//batch_size + 1}: {e}")
        
        # Process table content
        table_docs = []
        print("Processing table content...")
        for item in tqdm(processed_content["tables"], desc="Processing tables", unit="table"):
            try:
                # Tables might be large, chunk them too
                chunks = self._chunk_large_content(item["content"], max_chunk_size=1500)
                for chunk_idx, chunk in enumerate(chunks):
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "source": item["source"],
                            "page": item["page"],
                            "type": item["type"],
                            "chunk_index": chunk_idx,
                            "image_path": item.get("image_path", "")
                        }
                    )
                    table_docs.append(doc)
            except Exception as e:
                print(f"Error processing table from page {item['page']}: {e}")
        
        # Add table documents in batches
        if table_docs:
            batch_size = 30
            print(f"Adding {len(table_docs)} table chunks to vector store...")
            for i in tqdm(range(0, len(table_docs), batch_size), desc="Adding table batches"):
                batch = table_docs[i:i+batch_size]
                try:
                    self.table_store.add_documents(batch)
                except Exception as e:
                    print(f"Error adding table batch {i//batch_size + 1}: {e}")
        
        # Process visual content
        visual_docs = []
        print("Processing visual content...")
        for item in tqdm(processed_content["visuals"], desc="Processing visuals", unit="visual"):
            try:
                # Visual descriptions can also be large
                chunks = self._chunk_large_content(item["content"], max_chunk_size=1500)
                for chunk_idx, chunk in enumerate(chunks):
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "source": item["source"],
                            "page": item["page"],
                            "type": item["type"],
                            "chunk_index": chunk_idx,
                            "image_path": item.get("image_path", "")
                        }
                    )
                    visual_docs.append(doc)
            except Exception as e:
                print(f"Error processing visual from page {item['page']}: {e}")
        
        # Add visual documents in batches
        if visual_docs:
            batch_size = 30
            print(f"Adding {len(visual_docs)} visual chunks to vector store...")
            for i in tqdm(range(0, len(visual_docs), batch_size), desc="Adding visual batches"):
                batch = visual_docs[i:i+batch_size]
                try:
                    self.visual_store.add_documents(batch)
                except Exception as e:
                    print(f"Error adding visual batch {i//batch_size + 1}: {e}")
        
        print(f"✓ Successfully added {len(text_docs)} text, {len(table_docs)} table, and {len(visual_docs)} visual chunks")
    
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
