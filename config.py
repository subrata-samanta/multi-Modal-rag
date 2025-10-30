import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GOOGLE_CREDENTIALS_PATH = os.getenv("GOOGLE_CREDENTIALS_PATH", "key.json")
    
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    COLLECTION_NAME = "multimodal_rag"
    PERSIST_DIRECTORY = "./chroma_db"
    IMAGES_STORAGE_PATH = "./stored_images"  # Path to store original images
    
    # Gemini Vision prompts
    TEXT_EXTRACTION_PROMPT = """
    Analyze this image carefully and extract all readable text content. 
    Include:
    - All paragraphs and body text
    - Headings and subheadings
    - Bullet points and lists
    - Any textual annotations or labels
    
    Preserve the structure and formatting as much as possible.
    If there is no text content, respond with "No text content found."
    """
    
    TABLE_EXTRACTION_PROMPT = """
    Analyze this image and identify if there are any tables, data grids, or structured data present.
    
    If tables are found:
    1. Extract the complete table structure including headers, rows, and columns
    2. Convert each table to markdown format
    3. Provide a brief description of what the table contains
    
    Format your response as:
    ## Table Description: [brief description]
    [markdown table]
    
    If multiple tables exist, separate them clearly.
    If no tables are found, respond with "No tables found."
    """
    
    VISUAL ANALYSIS_PROMPT = """
    Analyze this image and describe any visual elements that convey information:
    
    1. **Charts and Graphs**: Describe type (bar, line, pie, etc.), axes, trends, and key insights
    2. **Diagrams and Flowcharts**: Explain the structure, flow, and relationships
    3. **Images and Illustrations**: Describe what they show and their relevance
    4. **Visual Patterns**: Note any significant visual information
    
    Focus on information that would help someone understand the content without seeing the image.
    If there are no significant visual elements, respond with "No visual elements found."
    """
