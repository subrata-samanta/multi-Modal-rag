from typing import List, Dict, Any
import os
from langchain_google_vertexai import ChatVertexAI
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from document_processor import DocumentProcessor
from vector_store import MultiModalVectorStore
from config import Config
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from tqdm import tqdm

class MultiModalRAGPipeline:
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.vector_store = MultiModalVectorStore()
        self._setup_llm()
    
    def _setup_llm(self):
        """Setup LLM with service account authentication"""
        try:
            # Set environment variable for Google Application Credentials
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = Config.GOOGLE_CREDENTIALS_PATH
            
            self.llm = ChatVertexAI(
                model_name="gemini-2.5-pro",
                temperature=0.1
            )
        except Exception as e:
            raise Exception(f"Failed to setup LLM: {e}")
    
    def ingest_document(self, file_path: str):
        """Ingest a document into the RAG pipeline"""
        print(f"Processing document: {file_path}")
        
        # Process document with parallel processing if enabled
        if Config.ENABLE_MULTIPROCESSING:
            processed_content = self.document_processor.process_document_parallel(file_path)
        else:
            processed_content = self.document_processor.process_document(file_path)
        
        # Add to vector stores
        self.vector_store.add_documents(processed_content)
        
        print("Document successfully ingested!")
    
    def ingest_document_with_logging(self, file_path: str, log_file, display_callback):
        """Ingest a document with page-by-page logging and display"""
        from document_processor import DocumentProcessor
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        print(f"Processing document: {file_path}")
        
        # Get document processor
        processor = self.document_processor
        
        # Convert document to images
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext == '.pdf':
            images = processor.convert_pdf_to_images(file_path)
        elif file_ext == '.pptx':
            images = processor.convert_pptx_to_images(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        processed_content = {
            "text": [],
            "tables": [],
            "visuals": []
        }
        
        if Config.ENABLE_MULTIPROCESSING and len(images) >= 2:
            # Parallel processing with logging
            print(f"Processing {len(images)} pages with {Config.MAX_WORKERS} workers...")
            
            # Prepare page data
            page_data_list = [
                {
                    'image': image,
                    'page_num': page_num + 1,
                    'file_path': file_path
                }
                for page_num, image in enumerate(images)
            ]
            
            # Process pages in parallel
            with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
                future_to_page = {
                    executor.submit(processor.process_page, page_data): page_data['page_num']
                    for page_data in page_data_list
                }
                
                results = {}
                for future in as_completed(future_to_page):
                    page_num = future_to_page[future]
                    try:
                        result = future.result()
                        results[result['page_num']] = result
                        print(f"✓ Completed page {result['page_num']}/{len(images)}")
                    except Exception as e:
                        print(f"✗ Error processing page {page_num}: {e}")
                        results[page_num] = {
                            'page_num': page_num,
                            'text': None,
                            'table': None,
                            'visual': None
                        }
            
            # Display and log results in page order
            for page_num in sorted(results.keys()):
                result = results[page_num]
                
                text_content = result['text']['content'] if result['text'] else ""
                table_content = result['table']['content'] if result['table'] else ""
                visual_content = result['visual']['content'] if result['visual'] else ""
                
                # Display and log
                display_callback(page_num, text_content, table_content, visual_content, log_file)
                
                # Collect processed content
                if result['text']:
                    processed_content["text"].append(result['text'])
                if result['table']:
                    processed_content["tables"].append(result['table'])
                if result['visual']:
                    processed_content["visuals"].append(result['visual'])
        else:
            # Sequential processing (original logic)
            for page_num, image in enumerate(images):
                print(f"\nProcessing page {page_num + 1}/{len(images)}...")
                
                # Extract text content
                text_content = processor.extract_content_from_image(image, "text")
                if text_content.strip() and "no text content found" not in text_content.lower():
                    processed_content["text"].append({
                        "content": text_content,
                        "page": page_num + 1,
                        "source": file_path,
                        "type": "text"
                    })
                
                # Extract table content
                table_content = processor.extract_content_from_image(image, "table")
                if table_content.strip() and "no tables found" not in table_content.lower():
                    processed_content["tables"].append({
                        "content": table_content,
                        "page": page_num + 1,
                        "source": file_path,
                        "type": "table"
                    })
                
                # Extract visual content
                visual_content = processor.extract_content_from_image(image, "visual")
                if visual_content.strip() and "no visual elements found" not in visual_content.lower():
                    processed_content["visuals"].append({
                        "content": visual_content,
                        "page": page_num + 1,
                        "source": file_path,
                        "type": "visual"
                    })
                
                # Display and log the extracted content for this page
                display_callback(page_num + 1, text_content, table_content, visual_content, log_file)
        
        # Add all processed content to vector stores
        self.vector_store.add_documents(processed_content)
        
        print("\nDocument successfully ingested!")
        
        # Return summary for logging
        summary = f"\n{'='*80}\n"
        summary += f"DOCUMENT PROCESSING SUMMARY\n"
        summary += f"{'='*80}\n"
        summary += f"Total Pages/Slides: {len(images)}\n"
        summary += f"Text Chunks: {len(processed_content['text'])}\n"
        summary += f"Table Entries: {len(processed_content['tables'])}\n"
        summary += f"Visual Descriptions: {len(processed_content['visuals'])}\n"
        summary += f"Processing Mode: {'Parallel' if Config.ENABLE_MULTIPROCESSING else 'Sequential'}\n"
        summary += f"{'='*80}\n"
        
        log_file.write(summary)
        return summary
    
    def ingest_document_with_parallel_logging(self, file_path: str, log_file, display_callback):
        """Ingest a document with parallel page processing and real-time progress"""
        from document_processor import DocumentProcessor
        
        print(f"Processing document: {file_path}")
        
        # Get document processor
        processor = self.document_processor
        
        # Convert document to images
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext == '.pdf':
            images = processor.convert_pdf_to_images(file_path)
        elif file_ext == '.pptx':
            images = processor.convert_pptx_to_images(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        total_pages = len(images)
        print(f"Total pages to process: {total_pages}")
        print(f"Using parallel processing with up to 3 workers...")
        
        processed_content = {
            "text": [],
            "tables": [],
            "visuals": []
        }
        
        # Create a lock for thread-safe operations
        lock = Lock()
        
        def process_page(page_num, image, pbar):
            """Process a single page"""
            try:
                # Extract all content types
                text_content = processor.extract_content_from_image(image, "text")
                table_content = processor.extract_content_from_image(image, "table")
                visual_content = processor.extract_content_from_image(image, "visual")
                
                # Prepare results
                page_results = {
                    "text": None,
                    "table": None,
                    "visual": None,
                    "page_num": page_num
                }
                
                if text_content.strip() and "no text content found" not in text_content.lower():
                    page_results["text"] = {
                        "content": text_content,
                        "page": page_num + 1,
                        "source": file_path,
                        "type": "text"
                    }
                
                if table_content.strip() and "no tables found" not in table_content.lower():
                    page_results["table"] = {
                        "content": table_content,
                        "page": page_num + 1,
                        "source": file_path,
                        "type": "table"
                    }
                
                if visual_content.strip() and "no visual elements found" not in visual_content.lower():
                    page_results["visual"] = {
                        "content": visual_content,
                        "page": page_num + 1,
                        "source": file_path,
                        "type": "visual"
                    }
                
                # Display and log with lock
                with lock:
                    display_callback(page_num + 1, text_content, table_content, visual_content, log_file, lock)
                
                # Update progress bar
                pbar.update(1)
                
                return page_results
            except Exception as e:
                pbar.update(1)
                raise e
        
        # Process pages in parallel with tqdm progress bar
        page_data = []
        
        # Create progress bar
        with tqdm(total=total_pages, desc="📄 Extracting Pages", unit="page", 
                 bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                 ncols=100) as pbar:
            
            with ThreadPoolExecutor(max_workers=3) as executor:
                # Submit all pages for processing
                futures = [
                    executor.submit(process_page, idx, img, pbar)
                    for idx, img in enumerate(images)
                ]
                
                # Collect results as they complete
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        page_data.append((result["page_num"], result))
                    except Exception as e:
                        tqdm.write(f"✗ Error processing page: {e}")
        
        print()  # New line after progress bar
        
        # Sort by page number and add to processed_content
        page_data.sort(key=lambda x: x[0])
        for page_num, result in page_data:
            if result["text"]:
                processed_content["text"].append(result["text"])
            if result["table"]:
                processed_content["tables"].append(result["table"])
            if result["visual"]:
                processed_content["visuals"].append(result["visual"])
        
        # Add all processed content to vector stores
        print("Adding extracted content to vector stores...")
        self.vector_store.add_documents(processed_content)
        
        print("✓ Document successfully ingested!")
        
        # Return summary for logging
        summary = f"\n{'='*80}\n"
        summary += f"DOCUMENT PROCESSING SUMMARY\n"
        summary += f"{'='*80}\n"
        summary += f"Total Pages/Slides: {total_pages}\n"
        summary += f"Text Chunks: {len(processed_content['text'])}\n"
        summary += f"Table Entries: {len(processed_content['tables'])}\n"
        summary += f"Visual Descriptions: {len(processed_content['visuals'])}\n"
        summary += f"Processing Mode: Parallel (3 workers)\n"
        summary += f"{'='*80}\n"
        
        log_file.write(summary)
        return summary
    
    def format_context(self, documents: List[Document]) -> str:
        """Format retrieved documents into context for the LLM"""
        context_parts = []
        
        # Group documents by type
        text_docs = [d for d in documents if d.metadata.get("type") == "text"]
        table_docs = [d for d in documents if d.metadata.get("type") == "table"]
        visual_docs = [d for d in documents if d.metadata.get("type") == "visual"]
        
        # Format text content
        if text_docs:
            context_parts.append("=== TEXT CONTENT ===")
            for doc in text_docs:
                context_parts.append(f"Page {doc.metadata.get('page', 'Unknown')}:")
                context_parts.append(doc.page_content)
                context_parts.append("")
        
        # Format table content
        if table_docs:
            context_parts.append("=== TABLE DATA ===")
            for doc in table_docs:
                context_parts.append(f"Page {doc.metadata.get('page', 'Unknown')}:")
                context_parts.append(doc.page_content)
                context_parts.append("")
        
        # Format visual content
        if visual_docs:
            context_parts.append("=== VISUAL INFORMATION ===")
            for doc in visual_docs:
                context_parts.append(f"Page {doc.metadata.get('page', 'Unknown')}:")
                context_parts.append(doc.page_content)
                context_parts.append("")
        
        return "\n".join(context_parts)
    
    def get_relevant_images(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Extract unique image paths from retrieved documents"""
        image_references = []
        seen_images = set()
        
        for doc in documents:
            image_path = doc.metadata.get("image_path", "")
            if image_path and image_path not in seen_images:
                seen_images.add(image_path)
                image_references.append({
                    "image_path": image_path,
                    "page": doc.metadata.get("page", "Unknown"),
                    "source": doc.metadata.get("source", "Unknown"),
                    "type": doc.metadata.get("type", "Unknown")
                })
        
        return image_references
    
    def extract_tables_from_context(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Extract table content and metadata from documents"""
        tables = []
        
        for doc in documents:
            if doc.metadata.get("type") == "table":
                tables.append({
                    "content": doc.page_content,
                    "page": doc.metadata.get("page", "Unknown"),
                    "source": doc.metadata.get("source", "Unknown"),
                    "image_path": doc.metadata.get("image_path", "")
                })
        
        return tables
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """Answer a question using the RAG pipeline"""
        # Retrieve relevant documents
        relevant_docs = self.vector_store.get_contextual_documents(question)
        
        if not relevant_docs:
            return {
                "answer": "I couldn't find relevant information to answer your question.",
                "sources": [],
                "context_used": "",
                "relevant_images": [],
                "relevant_tables": []
            }
        
        # Format context
        context = self.format_context(relevant_docs)
        
        # Create prompt with context
        prompt = f"""
        Based on the following context from documents, please answer the question.
        
        Context:
        {context}
        
        Question: {question}
        
        Instructions:
        - Use information from text, tables, and visual descriptions as needed
        - If the answer involves data from tables, present it clearly
        - If referencing visual information, explain it in detail
        - Cite which page or section the information comes from
        - If you cannot answer based on the provided context, say so clearly
        
        Answer:
        """
        
        # Generate answer using invoke method
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        # Extract source information
        sources = []
        for doc in relevant_docs:
            source_info = {
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "Unknown"),
                "type": doc.metadata.get("type", "Unknown")
            }
            if source_info not in sources:
                sources.append(source_info)
        
        # Extract answer content (strip any leading/trailing whitespace)
        answer_content = response.content.strip()
        
        return {
            "answer": answer_content,
            "sources": sources,
            "context_used": context,
            "relevant_images": self.get_relevant_images(relevant_docs),
            "relevant_tables": self.extract_tables_from_context(relevant_docs)
        }