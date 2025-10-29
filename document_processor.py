import os
from typing import List, Dict, Any
from PIL import Image
import pdf2image
from pptx import Presentation
import google.generativeai as genai
from langchain_google_vertexai import ChatVertexAI
from config import Config

class DocumentProcessor:
    def __init__(self):
        self._setup_authentication()
        self.vision_model = genai.GenerativeModel('gemini-pro-vision')
        self.llm = ChatVertexAI(
            model_name="gemini-pro-vision"
        )
        
    def _setup_authentication(self):
        """Setup Google authentication using service account key file"""
        try:
            # Set environment variable for Google Application Credentials
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = Config.GOOGLE_CREDENTIALS_PATH
            
            # Configure genai for vision model
            genai.configure()
            
        except Exception as e:
            raise Exception(f"Failed to setup Google authentication: {e}")
    
    def convert_pdf_to_images(self, pdf_path: str) -> List[Image.Image]:
        """Convert PDF pages to images"""
        try:
            images = pdf2image.convert_from_path(pdf_path)
            return images
        except Exception as e:
            print(f"Error converting PDF: {e}")
            return []
    
    def convert_pptx_to_images(self, pptx_path: str) -> List[Image.Image]:
        """Convert PPTX slides to images"""
        # Note: This is a simplified approach. For production, consider using 
        # libraries like python-pptx with additional image conversion tools
        print("PPTX to image conversion requires additional setup")
        return []
    
    def extract_content_from_image(self, image: Image.Image, content_type: str) -> str:
        """Extract specific content type from image using Gemini Vision"""
        try:
            if content_type == "text":
                prompt = Config.TEXT_EXTRACTION_PROMPT
            elif content_type == "table":
                prompt = Config.TABLE_EXTRACTION_PROMPT
            elif content_type == "visual":
                prompt = Config.VISUAL_ANALYSIS_PROMPT
            else:
                raise ValueError(f"Unknown content type: {content_type}")
            
            response = self.vision_model.generate_content([prompt, image])
            return response.text
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
