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
                model_name="text-embedding-005"
            )
        except Exception as e:
            raise Exception(f"Failed to setup embeddings: {e}")
    
    def _setup_query_analyzer(self):
        """Setup LLM for query analysis"""
        try:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = Config.GOOGLE_CREDENTIALS_PATH
            self.query_analyzer_llm = ChatVertexAI(
                model_name="gemini-2.5-pro",
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
    
    def _should_chunk_content(self, content: str, content_type: str, max_size: int = 15000) -> bool:
        """Determine if content should be chunked based on type and size"""
        # Tables and visuals should only be chunked if absolutely necessary
        if content_type in ["table", "visual"]:
            # Only chunk if content exceeds embedding model limits
            return len(content) > max_size
        return True
    
    def _preserve_table_structure(self, table_content: str) -> List[Dict[str, str]]:
        """Split tables intelligently while preserving structure"""
        # Strategy: Keep table header with each chunk
        lines = table_content.split('\n')
        
        # Find markdown table boundaries
        table_sections = []
        current_section = []
        in_table = False
        table_header = []
        
        for line in lines:
            if '|' in line and '---' in line:
                # This is a table separator
                in_table = True
                if current_section:
                    table_header = current_section[-1:]  # Store header
                current_section.append(line)
            elif '|' in line and in_table:
                current_section.append(line)
                # Check if section is getting too large
                if len('\n'.join(current_section)) > 3000:
                    table_sections.append({
                        'content': '\n'.join(current_section),
                        'has_header': True
                    })
                    # Start new section with header
                    current_section = table_header + [current_section[-1]]  # Keep header and separator
            elif in_table and not line.strip().startswith('|'):
                # End of table
                if current_section:
                    table_sections.append({
                        'content': '\n'.join(current_section),
                        'has_header': True
                    })
                in_table = False
                current_section = [line] if line.strip() else []
            else:
                current_section.append(line)
        
        # Add remaining content
        if current_section:
            table_sections.append({
                'content': '\n'.join(current_section),
                'has_header': True
            })
        
        return table_sections if table_sections else [{'content': table_content, 'has_header': False}]
    
    def _preserve_visual_structure(self, visual_content: str) -> List[Dict[str, str]]:
        """Split visual descriptions while preserving context"""
        # Strategy: Keep visual type and description together
        sections = []
        
        # Split by visual boundaries (### Visual X:)
        visual_blocks = []
        current_block = []
        
        lines = visual_content.split('\n')
        for line in lines:
            if line.startswith('### Visual'):
                if current_block:
                    visual_blocks.append('\n'.join(current_block))
                current_block = [line]
            else:
                current_block.append(line)
        
        if current_block:
            visual_blocks.append('\n'.join(current_block))
        
        # Each visual block is kept intact unless too large
        for block in visual_blocks:
            if len(block) > 4000:
                # Only split if absolutely necessary
                # Keep description and data summary together
                parts = block.split('**Key Insights**')
                if len(parts) == 2:
                    sections.append({
                        'content': parts[0] + '**Key Insights**',
                        'type': 'visual_main'
                    })
                    sections.append({
                        'content': '**Key Insights**' + parts[1],
                        'type': 'visual_insights'
                    })
                else:
                    sections.append({'content': block, 'type': 'visual_complete'})
            else:
                sections.append({'content': block, 'type': 'visual_complete'})
        
        return sections if sections else [{'content': visual_content, 'type': 'visual_complete'}]
    
    def add_documents(self, processed_content: Dict[str, List[Dict[str, Any]]]):
        """Add processed documents with intelligent preservation of tables and visuals"""
        from tqdm import tqdm
        
        # Process text content with semantic/intelligent chunking
        text_docs = []
        print(f"Processing text content with {self.chunking_strategy} chunking strategy...")
        for item in tqdm(processed_content["text"], desc="Chunking text", unit="page"):
            try:
                # Use intelligent chunking for text
                chunks = self._chunk_text_with_strategy(item["content"])
                
                for chunk_idx, chunk in enumerate(chunks):
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "source": item["source"],
                            "page": item["page"],
                            "type": item["type"],
                            "chunk_index": chunk_idx,
                            "total_chunks": len(chunks),
                            "chunking_strategy": self.chunking_strategy,
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
        
        # Process table content - PRESERVE STRUCTURE
        table_docs = []
        print("Processing table content with structure preservation...")
        for item in tqdm(processed_content["tables"], desc="Processing tables", unit="table"):
            try:
                content = item["content"]
                
                # Check if chunking is necessary
                if self._should_chunk_content(content, "table"):
                    # Preserve table structure when chunking
                    table_sections = self._preserve_table_structure(content)
                    
                    for section_idx, section in enumerate(table_sections):
                        doc = Document(
                            page_content=section['content'],
                            metadata={
                                "source": item["source"],
                                "page": item["page"],
                                "type": item["type"],
                                "chunk_index": section_idx,
                                "total_chunks": len(table_sections),
                                "has_header": section.get('has_header', False),
                                "is_complete_table": len(table_sections) == 1,
                                "chunking_strategy": "structure_preserved",
                                "image_path": item.get("image_path", "")
                            }
                        )
                        table_docs.append(doc)
                else:
                    # Keep table intact - NO CHUNKING
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": item["source"],
                            "page": item["page"],
                            "type": item["type"],
                            "chunk_index": 0,
                            "total_chunks": 1,
                            "has_header": True,
                            "is_complete_table": True,
                            "chunking_strategy": "intact",
                            "image_path": item.get("image_path", "")
                        }
                    )
                    table_docs.append(doc)
                    
            except Exception as e:
                print(f"Error processing table from page {item['page']}: {e}")
        
        # Add table documents in batches
        if table_docs:
            batch_size = 30
            print(f"Adding {len(table_docs)} table entries to vector store...")
            for i in tqdm(range(0, len(table_docs), batch_size), desc="Adding table batches"):
                batch = table_docs[i:i+batch_size]
                try:
                    self.table_store.add_documents(batch)
                except Exception as e:
                    print(f"Error adding table batch {i//batch_size + 1}: {e}")
        
        # Process visual content - PRESERVE CONTEXT
        visual_docs = []
        print("Processing visual content with context preservation...")
        for item in tqdm(processed_content["visuals"], desc="Processing visuals", unit="visual"):
            try:
                content = item["content"]
                
                # Check if chunking is necessary
                if self._should_chunk_content(content, "visual"):
                    # Preserve visual context when chunking
                    visual_sections = self._preserve_visual_structure(content)
                    
                    for section_idx, section in enumerate(visual_sections):
                        doc = Document(
                            page_content=section['content'],
                            metadata={
                                "source": item["source"],
                                "page": item["page"],
                                "type": item["type"],
                                "chunk_index": section_idx,
                                "total_chunks": len(visual_sections),
                                "visual_type": section.get('type', 'unknown'),
                                "is_complete_visual": len(visual_sections) == 1,
                                "chunking_strategy": "context_preserved",
                                "image_path": item.get("image_path", "")
                            }
                        )
                        visual_docs.append(doc)
                else:
                    # Keep visual intact - NO CHUNKING
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": item["source"],
                            "page": item["page"],
                            "type": item["type"],
                            "chunk_index": 0,
                            "total_chunks": 1,
                            "visual_type": "complete",
                            "is_complete_visual": True,
                            "chunking_strategy": "intact",
                            "image_path": item.get("image_path", "")
                        }
                    )
                    visual_docs.append(doc)
                    
            except Exception as e:
                print(f"Error processing visual from page {item['page']}: {e}")
        
        # Add visual documents in batches
        if visual_docs:
            batch_size = 30
            print(f"Adding {len(visual_docs)} visual entries to vector store...")
            for i in tqdm(range(0, len(visual_docs), batch_size), desc="Adding visual batches"):
                batch = visual_docs[i:i+batch_size]
                try:
                    self.visual_store.add_documents(batch)
                except Exception as e:
                    print(f"Error adding visual batch {i//batch_size + 1}: {e}")
        
        print(f"✓ Successfully added:")
        print(f"  - {len(text_docs)} text chunks")
        print(f"  - {len(table_docs)} table entries ({sum(1 for d in table_docs if d.metadata.get('is_complete_table'))} complete)")
        print(f"  - {len(visual_docs)} visual entries ({sum(1 for d in visual_docs if d.metadata.get('is_complete_visual'))} complete)")
    
    def search_relevant_content(self, query: str, k: int = 3) -> Dict[str, List[Document]]:
        """Search with special handling for complete tables/visuals"""
        # Retrieve more results initially to check for complete items
        text_results = self.text_store.similarity_search(query, k=k)
        table_results = self.table_store.similarity_search(query, k=k*2)
        visual_results = self.visual_store.similarity_search(query, k=k*2)
        
        # Prioritize complete tables and visuals
        complete_tables = [doc for doc in table_results if doc.metadata.get('is_complete_table', False)]
        chunked_tables = [doc for doc in table_results if not doc.metadata.get('is_complete_table', False)]
        
        complete_visuals = [doc for doc in visual_results if doc.metadata.get('is_complete_visual', False)]
        chunked_visuals = [doc for doc in visual_results if not doc.metadata.get('is_complete_visual', False)]
        
        # Prefer complete items, then fill with chunks
        final_tables = complete_tables[:k]
        if len(final_tables) < k:
            final_tables.extend(chunked_tables[:k - len(final_tables)])
        
        final_visuals = complete_visuals[:k]
        if len(final_visuals) < k:
            final_visuals.extend(chunked_visuals[:k - len(final_visuals)])
        
        # If chunked items are retrieved, get all related chunks
        final_tables = self._gather_related_chunks(final_tables, self.table_store)
        final_visuals = self._gather_related_chunks(final_visuals, self.visual_store)
        
        return {
            "text": text_results,
            "tables": final_tables[:k*2],  # Allow more for reconstructed tables
            "visuals": final_visuals[:k*2]
        }
    
    def _gather_related_chunks(self, documents: List[Document], store) -> List[Document]:
        """Gather all chunks belonging to the same table/visual"""
        result = []
        seen_sources = set()
        
        for doc in documents:
            if doc.metadata.get('is_complete_table') or doc.metadata.get('is_complete_visual'):
                # Already complete, add as is
                result.append(doc)
            else:
                # This is a chunk, get all related chunks
                source_key = (doc.metadata.get('source'), doc.metadata.get('page'), doc.metadata.get('type'))
                
                if source_key not in seen_sources:
                    seen_sources.add(source_key)
                    
                    # Search for all chunks from same source
                    total_chunks = doc.metadata.get('total_chunks', 1)
                    if total_chunks > 1:
                        # Retrieve all chunks (this is a simplified approach)
                        # In production, you'd want a more efficient method
                        all_chunks = store.similarity_search(
                            doc.page_content, 
                            k=total_chunks * 2,
                            filter={"page": doc.metadata.get('page')}
                        )
                        
                        # Filter to exact matches
                        related = [
                            c for c in all_chunks 
                            if (c.metadata.get('source'), c.metadata.get('page'), c.metadata.get('type')) == source_key
                        ]
                        
                        # Sort by chunk index
                        related.sort(key=lambda x: x.metadata.get('chunk_index', 0))
                        result.extend(related)
                    else:
                        result.append(doc)
        
        return result
    
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
