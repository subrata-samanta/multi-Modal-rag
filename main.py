import os
from rag_pipeline import MultiModalRAGPipeline

def main():
    # Initialize the RAG pipeline
    rag = MultiModalRAGPipeline()
    
    print("=== Multimodal RAG Pipeline ===")
    print("This pipeline can process PDF and PPTX files using Google Gemini Vision")
    print("It extracts text, tables, and visual information for comprehensive Q&A")
    print()
    
    while True:
        print("\n1. Ingest Document")
        print("2. Ask Question")
        print("3. Document Summary")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            file_path = input("Enter the path to your PDF or PPTX file: ").strip()
            
            if not os.path.exists(file_path):
                print("File not found. Please check the path.")
                continue
            
            if not file_path.lower().endswith(('.pdf', '.pptx')):
                print("Unsupported file type. Please use PDF or PPTX files.")
                continue
            
            try:
                rag.ingest_document(file_path)
            except Exception as e:
                print(f"Error processing document: {e}")
        
        elif choice == "2":
            question = input("Enter your question: ").strip()
            
            if not question:
                print("Please enter a valid question.")
                continue
            
            print("\nProcessing your question...")
            try:
                result = rag.answer_question(question)
                
                print(f"\n=== ANSWER ===")
                print(result["answer"])
                
                print(f"\n=== SOURCES ===")
                for source in result["sources"]:
                    print(f"- {source['source']} (Page {source['page']}, Type: {source['type']})")
                
                print(f"\n=== RETRIEVAL INFO ===")
                print(f"Documents retrieved: {result['num_documents_retrieved']}")
                
            except Exception as e:
                print(f"Error answering question: {e}")
        
        elif choice == "3":
            try:
                summary = rag.get_document_summary()
                print(f"\n=== DOCUMENT SUMMARY ===")
                print(f"Text chunks: {summary['text_chunks']}")
                print(f"Table entries: {summary['table_entries']}")
                print(f"Visual descriptions: {summary['visual_descriptions']}")
                print(f"Total entries: {summary['total_entries']}")
            except Exception as e:
                print(f"Error getting summary: {e}")
        
        elif choice == "4":
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")

if __name__ == "__main__":
    main()
