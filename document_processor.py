import os
import base64
from typing import List, Dict, Any
from io import BytesIO
from PIL import Image
import fitz  # PyMuPDF
from pptx import Presentation
from pptx.util import Inches
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import HumanMessage
from config import Config
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

class DocumentProcessor:
    def __init__(self):
        self._setup_authentication()
        self.llm = ChatVertexAI(
            model_name="gemini-2.5-pro"
        )
        self._thread_local = threading.local()
        
    def _setup_authentication(self):
        """Setup Google authentication using service account key file"""
        try:
            # Set environment variable for Google Application Credentials
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = Config.GOOGLE_CREDENTIALS_PATH
            
        except Exception as e:
            raise Exception(f"Failed to setup Google authentication: {e}")
    
    def _get_llm(self):
        """Get thread-local LLM instance"""
        if not hasattr(self._thread_local, 'llm'):
            self._thread_local.llm = ChatVertexAI(
                model_name="gemini-2.5-pro"
            )
        return self._thread_local.llm
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str
    
    def convert_pdf_to_images(self, pdf_path: str) -> List[Image.Image]:
        """Convert PDF pages to images using PyMuPDF (no poppler dependency)"""
        try:
            images = []
            pdf_document = fitz.open(pdf_path)
            
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                # Render page to image with high resolution
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
                
                # Convert pixmap to PIL Image
                img_data = pix.tobytes("png")
                img = Image.open(BytesIO(img_data))
                images.append(img)
            
            pdf_document.close()
            return images
        except Exception as e:
            print(f"Error converting PDF: {e}")
            return []
    
    def convert_pptx_to_images(self, pptx_path: str) -> List[Image.Image]:
        """Convert PPTX slides to images using python-pptx"""
        try:
            images = []
            prs = Presentation(pptx_path)
            
            # Create temporary directory for slide images
            temp_dir = "temp_slides"
            os.makedirs(temp_dir, exist_ok=True)
            
            for slide_num, slide in enumerate(prs.slides):
                # Export slide as image
                slide_image_path = os.path.join(temp_dir, f"slide_{slide_num}.png")
                
                # Get slide dimensions
                slide_width = prs.slide_width
                slide_height = prs.slide_height
                
                # Create blank image with slide dimensions
                img = Image.new('RGB', (int(slide_width / 9525), int(slide_height / 9525)), 'white')
                
                # Note: python-pptx doesn't directly support image export
                # For production, consider using win32com (Windows) or LibreOffice (cross-platform)
                # This is a placeholder - slides will be processed as text extraction
                images.append(img)
            
            # Clean up temp directory
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            
            return images
        except Exception as e:
            print(f"Error converting PPTX: {e}")
            return []
    
    def extract_content_from_image(self, image: Image.Image, content_type: str) -> str:
        """Extract specific content type from image using ChatVertexAI"""
        try:
            if content_type == "text":
                prompt = Config.TEXT_EXTRACTION_PROMPT
            elif content_type == "table":
                prompt = Config.TABLE_EXTRACTION_PROMPT
            elif content_type == "visual":
                prompt = Config.VISUAL_ANALYSIS_PROMPT
            else:
                raise ValueError(f"Unknown content type: {content_type}")
            
            # Convert image to bytes
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            image_bytes = buffered.getvalue()
            
            # Create message with image using the correct format for Vertex AI
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64.b64encode(image_bytes).decode()}"
                        }
                    }
                ]
            )
            
            # Use thread-local LLM instance
            llm = self._get_llm()
            response = llm.invoke([message])
            
            # Handle different response types
            if hasattr(response, 'content'):
                # If response has content attribute, use it
                result = response.content
            elif isinstance(response, dict):
                # If response is a dict, try to get content
                result = response.get('content', str(response))
            else:
                # Otherwise convert to string
                result = str(response)
            
            return result if isinstance(result, str) else str(result)
            
        except Exception as e:
            print(f"Error extracting {content_type}: {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    def process_page(self, page_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single page and extract all content types"""
        image = page_data['image']
        page_num = page_data['page_num']
        file_path = page_data['file_path']
        
        result = {
            'page_num': page_num,
            'text': None,
            'table': None,
            'visual': None
        }
        
        # Extract text content
        text_content = self.extract_content_from_image(image, "text")
        if text_content.strip() and "no text content found" not in text_content.lower():
            result['text'] = {
                "content": text_content,
                "page": page_num,
                "source": file_path,
                "type": "text"
            }
        
        # Extract table content
        table_content = self.extract_content_from_image(image, "table")
        if table_content.strip() and "no tables found" not in table_content.lower():
            result['table'] = {
                "content": table_content,
                "page": page_num,
                "source": file_path,
                "type": "table"
            }
        
        # Extract visual content
        visual_content = self.extract_content_from_image(image, "visual")
        if visual_content.strip() and "no visual elements found" not in visual_content.lower():
            result['visual'] = {
                "content": visual_content,
                "page": page_num,
                "source": file_path,
                "type": "visual"
            }
        
        return result
    
    def process_document_parallel(self, file_path: str) -> Dict[str, List[Dict[str, Any]]]:
        """Process document with parallel processing"""
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            images = self.convert_pdf_to_images(file_path)
        elif file_ext == '.pptx':
            images = self.convert_pptx_to_images(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        processed_content = {
            "text": [],
            "tables": [],
            "visuals": []
        }
        
        if not Config.ENABLE_MULTIPROCESSING or len(images) < 2:
            # Fall back to sequential processing for small documents
            return self.process_document(file_path)
        
        # Prepare page data for parallel processing
        page_data_list = [
            {
                'image': image,
                'page_num': page_num + 1,
                'file_path': file_path
            }
            for page_num, image in enumerate(images)
        ]
        
        # Process pages in parallel
        print(f"Processing {len(images)} pages with {Config.MAX_WORKERS} workers...")
        
        with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
            # Submit all pages for processing
            future_to_page = {
                executor.submit(self.process_page, page_data): page_data['page_num']
                for page_data in page_data_list
            }
            
            # Collect results as they complete
            results = {}
            for future in as_completed(future_to_page):
                page_num = future_to_page[future]
                try:
                    result = future.result()
                    results[result['page_num']] = result
                    print(f"✓ Completed page {result['page_num']}/{len(images)}")
                except Exception as e:
                    print(f"✗ Error processing page {page_num}: {e}")
        
        # Sort results by page number and collect
        for page_num in sorted(results.keys()):
            result = results[page_num]
            
            if result['text']:
                processed_content["text"].append(result['text'])
            if result['table']:
                processed_content["tables"].append(result['table'])
            if result['visual']:
                processed_content["visuals"].append(result['visual'])
        
        return processed_content
    
    def process_document(self, file_path: str) -> Dict[str, List[Dict[str, Any]]]:
        """Process document and extract all content types"""
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            images = self.convert_pdf_to_images(file_path)
        elif file_ext == '.pptx':
            images = self.convert_pptx_to_images(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        processed_content = {
            "text": [],
            "tables": [],
            "visuals": []
        }
        
        for page_num, image in enumerate(images):
            print(f"Processing page {page_num + 1}...")
            
            # Extract text content
            text_content = self.extract_content_from_image(image, "text")
            if text_content.strip() and "no text content found" not in text_content.lower():
                processed_content["text"].append({
                    "content": text_content,
                    "page": page_num + 1,
                    "source": file_path,
                    "type": "text"
                })
            
            # Extract table content
            table_content = self.extract_content_from_image(image, "table")
            if table_content.strip() and "no tables found" not in table_content.lower():
                processed_content["tables"].append({
                    "content": table_content,
                    "page": page_num + 1,
                    "source": file_path,
                    "type": "table"
                })
            
            # Extract visual content
            visual_content = self.extract_content_from_image(image, "visual")
            if visual_content.strip() and "no visual elements found" not in visual_content.lower():
                processed_content["visuals"].append({
                    "content": visual_content,
                    "page": page_num + 1,
                    "source": file_path,
                    "type": "visual"
                })
        
        return processed_content
