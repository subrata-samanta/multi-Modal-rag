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

class DocumentProcessor:
    def __init__(self):
        self._setup_authentication()
        self.llm = ChatVertexAI(
            model_name="gemini-2.0-flash-exp"
        )
        
    def _setup_authentication(self):
        """Setup Google authentication using service account key file"""
        try:
            # Set environment variable for Google Application Credentials
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = Config.GOOGLE_CREDENTIALS_PATH
            
        except Exception as e:
            raise Exception(f"Failed to setup Google authentication: {e}")
    
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
            
            # Convert image to base64
            image_base64 = self._image_to_base64(image)
            
            # Create message with image
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": f"data:image/png;base64,{image_base64}"
                    }
                ]
            )
            
            response = self.llm.invoke([message])
            return response.content
        except Exception as e:
            print(f"Error extracting {content_type}: {e}")
            return ""
    
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
            if text_content.strip():
                processed_content["text"].append({
                    "content": text_content,
                    "page": page_num + 1,
                    "source": file_path,
                    "type": "text"
                })
            
            # Extract table content
            table_content = self.extract_content_from_image(image, "table")
            if table_content.strip() and "table" in table_content.lower():
                processed_content["tables"].append({
                    "content": table_content,
                    "page": page_num + 1,
                    "source": file_path,
                    "type": "table"
                })
            
            # Extract visual content
            visual_content = self.extract_content_from_image(image, "visual")
            if visual_content.strip():
                processed_content["visuals"].append({
                    "content": visual_content,
                    "page": page_num + 1,
                    "source": file_path,
                    "type": "visual"
                })
        
        return processed_content
