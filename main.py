import os
from rag_pipeline import MultiModalRAGPipeline
from datetime import datetime

def save_extraction_log(file_path, content, output_folder="extraction_logs", is_error=False):
    """Save extracted content to a text file for review"""
    # Create output folder if it doesn't exist
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, output_folder)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a sanitized filename
    original_filename = os.path.splitext(os.path.basename(file_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    status_prefix = "ERROR_" if is_error else ""
    output_filename = f"{status_prefix}{original_filename}_{timestamp}.txt"
    output_path = os.path.join(output_dir, output_filename)
    
    # Write content to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"{'='*80}\n")
        f.write(f"EXTRACTION LOG {'[ERROR]' if is_error else '[SUCCESS]'}\n")
        f.write(f"{'='*80}\n")
        f.write(f"Source File: {file_path}\n")
        f.write(f"Extraction Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*80}\n\n")
        f.write(content)
    
    return output_path

def ingest_mode(rag):
    """Handle document/folder ingestion"""
    print("\n=== INGESTION MODE ===")
    print("1. Ingest Single Document")
    print("2. Ingest Folder")
    print("3. Back to Main Menu")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        file_path = input("Enter the path to your PDF or PPTX file: ").strip()
        
        if not os.path.exists(file_path):
            print("File not found. Please check the path.")
            return
        
        if not file_path.lower().endswith(('.pdf', '.pptx')):
            print("Unsupported file type. Please use PDF or PPTX files.")
            return
        
        try:
            print(f"Processing document: {file_path}")
            extracted_content = rag.ingest_document(file_path)
            
            # Ensure content is a string
            if extracted_content is None:
                extracted_content = "No content returned from ingestion process."
            elif not isinstance(extracted_content, str):
                extracted_content = str(extracted_content)
            
            # Save extraction log
            log_path = save_extraction_log(file_path, extracted_content, is_error=False)
            
            print("✓ Document ingested successfully!")
            print(f"✓ Extraction log saved to: {log_path}")
        except Exception as e:
            error_msg = str(e)
            
            # Create error log
            error_content = f"ERROR DETAILS:\n{'-'*80}\n"
            error_content += f"Error Type: {type(e).__name__}\n"
            error_content += f"Error Message: {error_msg}\n"
            error_content += f"\nFull Traceback:\n{'-'*80}\n"
            
            import traceback
            error_content += traceback.format_exc()
            
            # Save error log
            log_path = save_extraction_log(file_path, error_content, is_error=True)
            
            if "zlib error" in error_msg or "incorrect header" in error_msg:
                print(f"✗ Error: The PDF file appears to be corrupted or has an invalid format.")
                print(f"  Please verify the file integrity or try re-downloading/re-saving the PDF.")
            elif "MuPDF error" in error_msg:
                print(f"✗ Error: Unable to read the PDF file. The file may be corrupted or password-protected.")
            else:
                print(f"✗ Error processing document: {e}")
            
            print(f"✗ Error log saved to: {log_path}")
    
    elif choice == "2":
        folder_path = input("Enter the path to the folder containing PDF/PPTX files: ").strip()
        
        if not os.path.exists(folder_path):
            print("Folder not found. Please check the path.")
            return
        
        if not os.path.isdir(folder_path):
            print("The provided path is not a folder.")
            return
        
        try:
            # Get all PDF and PPTX files in the folder
            files = [f for f in os.listdir(folder_path) 
                    if f.lower().endswith(('.pdf', '.pptx'))]
            
            if not files:
                print("No PDF or PPTX files found in the folder.")
                return
            
            print(f"\nFound {len(files)} file(s) to process:")
            for f in files:
                print(f"  - {f}")
            
            confirm = input("\nProceed with ingestion? (y/n): ").strip().lower()
            if confirm != 'y':
                print("Folder ingestion cancelled.")
                return
            
            successful = 0
            failed = 0
            corrupted_files = []
            
            for filename in files:
                file_path = os.path.join(folder_path, filename)
                try:
                    print(f"\nProcessing: {filename}...")
                    extracted_content = rag.ingest_document(file_path)
                    
                    # Ensure content is a string
                    if extracted_content is None:
                        extracted_content = "No content returned from ingestion process."
                    elif not isinstance(extracted_content, str):
                        extracted_content = str(extracted_content)
                    
                    # Save extraction log
                    log_path = save_extraction_log(file_path, extracted_content, is_error=False)
                    
                    print(f"✓ Successfully ingested: {filename}")
                    print(f"  Log saved: {os.path.basename(log_path)}")
                    successful += 1
                except Exception as e:
                    error_msg = str(e)
                    
                    # Create error log
                    error_content = f"ERROR DETAILS:\n{'-'*80}\n"
                    error_content += f"Error Type: {type(e).__name__}\n"
                    error_content += f"Error Message: {error_msg}\n"
                    error_content += f"\nFull Traceback:\n{'-'*80}\n"
                    
                    import traceback
                    error_content += traceback.format_exc()
                    
                    # Save error log
                    log_path = save_extraction_log(file_path, error_content, is_error=True)
                    
                    if "zlib error" in error_msg or "incorrect header" in error_msg or "MuPDF error" in error_msg:
                        print(f"✗ Failed: {filename} (corrupted or invalid format)")
                        corrupted_files.append(filename)
                    else:
                        print(f"✗ Failed to ingest {filename}: {e}")
                    
                    print(f"  Error log saved: {os.path.basename(log_path)}")
                    failed += 1
            
            print(f"\n=== Ingestion Complete ===")
            print(f"Successfully ingested: {successful} file(s)")
            print(f"Failed: {failed} file(s)")
            if corrupted_files:
                print(f"\nCorrupted/Invalid files:")
                for cf in corrupted_files:
                    print(f"  - {cf}")
            
            # Show extraction logs location
            base_dir = os.path.dirname(os.path.abspath(__file__))
            logs_dir = os.path.join(base_dir, "extraction_logs")
            print(f"\nExtraction logs saved to: {logs_dir}")
            
        except Exception as e:
            print(f"Error processing folder: {e}")

def inference_mode(rag):
    """Handle question answering with mode selection"""
    print("\n=== INFERENCE MODE ===")
    print("Select inference type:")
    print("1. Auto (intelligent selection)")
    print("2. Text-only (faster)")
    print("3. Multimodal (comprehensive)")
    print("4. Back to Main Menu")
    
    mode_choice = input("\nEnter your choice (1-4): ").strip()
    
    if mode_choice == "4":
        return
    
    mode_map = {"1": "auto", "2": "text", "3": "multimodal"}
    inference_type = mode_map.get(mode_choice)
    
    if not inference_type:
        print("Invalid choice.")
        return
    
    print(f"\nInference mode: {inference_type.upper()}")
    question = input("Enter your question: ").strip()
    
    if not question:
        print("Please enter a valid question.")
        return
    
    print("\nProcessing your question...")
    try:
        result = rag.answer_question(question, mode=inference_type)
        
        print(f"\n=== ANSWER ===")
        print(result["answer"])
        
        print(f"\n=== SOURCES ===")
        for source in result["sources"]:
            print(f"- {source['source']} (Page {source['page']}, Type: {source['type']})")
        
        print(f"\n=== RETRIEVAL INFO ===")
        print(f"Documents retrieved: {result['num_documents_retrieved']}")
        
    except Exception as e:
        print(f"Error answering question: {e}")

def main():
    # Initialize the RAG pipeline
    rag = MultiModalRAGPipeline()
    
    print("=== Multimodal RAG Pipeline ===")
    print("This pipeline can process PDF and PPTX files using Google Gemini Vision")
    print("It extracts text, tables, and visual information for comprehensive Q&A")
    print()
    
    while True:
        print("\n" + "="*50)
        print("MAIN MENU")
        print("="*50)
        print("1. Ingestion (Add documents)")
        print("2. Inference (Ask questions)")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            ingest_mode(rag)
        
        elif choice == "2":
            inference_mode(rag)
        
        elif choice == "3":
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
