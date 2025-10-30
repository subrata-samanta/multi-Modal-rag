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
    Extract all readable text content from this image.
    Focus on paragraphs, headings, bullet points, and any textual information.
    Do NOT extract tables or visual elements here - only pure text content.
    Preserve the structure and formatting as much as possible.
    If there is no text content, respond with "No text content found."
    """
    
    TABLE_EXTRACTION_PROMPT = """
    Analyze this image for any tables or tabular data.

    If tables are present, for EACH table provide:

    1. **Table Description**: A brief description of what the table represents
    2. **Table Data**: Extract the table in Markdown format with proper alignment
       - Use | for column separators
       - Use |---|---| for header separators
       - Preserve all rows and columns accurately
    3. **Key Insights**: List 2-3 key insights or notable data points from the table

    Format your response as:

    ### Table 1: [Brief Title]
    **Description**: [What this table shows]

    **Table Data**:
    | Column 1 | Column 2 | Column 3 |
    |----------|----------|----------|
    | Data 1   | Data 2   | Data 3   |
    | Data 4   | Data 5   | Data 6   |

    **Key Insights**:
    - Insight 1
    - Insight 2
    - Insight 3

    If there are multiple tables, repeat this format for each table.
    If NO tables are present, respond with "No tables found."
    """
    
    VISUAL_ANALYSIS_PROMPT = """
    Analyze this image for any visual elements such as charts, graphs, diagrams, images, or illustrations.
    Do NOT analyze text or tables - focus only on visual/graphical content.

    If visual elements are present, for EACH visual element provide:

    1. **Visual Type**: Identify the type (e.g., Bar Chart, Pie Chart, Line Graph, Diagram, Flowchart, Image, Icon, etc.)
    2. **Visual Description**: Describe what the visual shows in detail
    3. **Data Summary**: If it's a chart/graph, extract the data in a tabular format:

    | Category/Label | Value/Metric | Additional Info |
    |----------------|--------------|-----------------|
    | Item 1         | Value 1      | Detail 1        |
    | Item 2         | Value 2      | Detail 2        |

    4. **Key Insights**: List 2-3 key insights, trends, or important observations from the visual

    Format your response as:

    ### Visual 1: [Type] - [Brief Title]
    **Description**: [Detailed description of what this visual represents]

    **Data Summary**:
    | Category | Value | Notes |
    |----------|-------|-------|
    | Data 1   | Val 1 | Note 1|
    | Data 2   | Val 2 | Note 2|

    **Key Insights**:
    - Insight 1: [Observation or trend]
    - Insight 2: [Pattern or comparison]
    - Insight 3: [Notable finding]

    **Additional Context**: [Any other relevant information about colors, legends, annotations, etc.]

    If there are multiple visual elements, repeat this format for each one.
    If NO visual elements are present, respond with "No visual elements found."
    """
    
    # Parallel Processing Configuration
    MAX_WORKERS = 4  # Number of parallel workers for processing pages
    BATCH_SIZE = 3   # Number of pages to process together in each batch
    ENABLE_MULTIPROCESSING = True  # Set to False to disable parallel processing
