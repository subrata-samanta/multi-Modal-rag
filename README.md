# Multimodal RAG Pipeline with Google Gemini Vision

A comprehensive Retrieval-Augmented Generation (RAG) pipeline that processes PDF and PPTX documents using Google Gemini Vision model to extract text, tables, and visual information for intelligent question answering.

## Features

- **Multimodal Document Processing**: Handles PDF and PPTX files using vision models
- **No External Dependencies**: Uses PyMuPDF for PDF processing (no poppler required)
- **Structured Content Extraction**: Separates text, tables, and visual information
- **LLM-Driven Retrieval**: Uses Gemini to analyze query intent and optimize retrieval
- **Comprehensive Q&A**: Answers questions using text, tabular, and visual data
- **Clean Architecture**: Modular design for easy maintenance and extension

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. **Google Cloud Service Account Setup**:
   - Go to Google Cloud Console
   - Create a new project or select an existing one
   - Enable the Vertex AI API and Generative AI API
   - Create a service account with appropriate permissions:
     - Vertex AI User
     - Service Account Token Creator
   - Download the service account key as JSON
   - Place the key file as `key.json` in the project root directory

3. Update the `.env` file if your key file has a different name:
```
GOOGLE_CREDENTIALS_PATH=your_key_file.json
```

4. Run the application:
```bash
python main.py
```

## Key Dependencies

- **PyMuPDF (fitz)**: PDF processing without external dependencies (no poppler needed)
- **python-pptx**: PPTX file handling
- **LangChain**: Framework for LLM applications
- **Chroma**: Vector database for embeddings
- **Google Vertex AI**: Gemini models for vision and text processing

## Required Google Cloud APIs

Make sure to enable these APIs in your Google Cloud project:
- Vertex AI API
- Generative AI API

## File Structure

```
c:\Users\Subrata Samanta\Desktop\multi Modal rag\
├── key.json                 # Google service account key (not in repo)
├── .env                     # Environment configuration
├── config.py               # Configuration settings
├── document_processor.py   # Document processing module
├── vector_store.py         # Vector store management
├── rag_pipeline.py         # Main RAG pipeline
├── main.py                 # Application entry point
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Security Note

- Never commit your `key.json` file to version control
- Add `key.json` to your `.gitignore` file
- Ensure proper IAM permissions for your service account

## Architecture

- **DocumentProcessor**: Converts documents to images and extracts content using Gemini Vision
- **MultiModalVectorStore**: Manages separate vector stores for different content types
- **MultiModalRAGPipeline**: Orchestrates the entire pipeline with optimized retrieval

## Usage

1. **Ingest Documents**: Add PDF or PPTX files to the knowledge base
2. **Ask Questions**: Query the system using natural language
3. **Get Answers**: Receive comprehensive answers with source citations

## Optimization Features

- **Adaptive Retrieval**: Adjusts retrieval strategy based on question type
- **Content Type Separation**: Stores text, tables, and visuals separately for better organization
- **Context Formatting**: Structures retrieved content for optimal LLM processing
- **Source Tracking**: Maintains metadata for proper citation and traceability
