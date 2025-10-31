import os
from rag_pipeline import MultiModalRAGPipeline
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from session_manager import SessionManager, ChatSession

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

def display_and_log_page_content(page_num, text_content, table_content, visual_content, log_file, lock=None):
    """Display and log extracted content for a single page"""
    output = []
    output.append(f"\n{'='*80}")
    output.append(f"PAGE {page_num} EXTRACTION RESULTS")
    output.append(f"{'='*80}")
    
    # Prepare text content
    if text_content and text_content.strip() and "no text content found" not in text_content.lower():
        output.append(f"\n--- TEXT CONTENT ---")
        output.append(text_content[:500] + "..." if len(text_content) > 500 else text_content)
        log_text = f"\n{'='*80}\nPAGE {page_num} - TEXT CONTENT\n{'='*80}\n{text_content}\n\n"
    else:
        output.append(f"\n--- TEXT CONTENT ---")
        output.append("No text content found on this page.")
        log_text = f"\n{'='*80}\nPAGE {page_num} - TEXT CONTENT\n{'='*80}\nNo text content found on this page.\n\n"
    
    # Prepare table content
    if table_content and table_content.strip() and "no tables found" not in table_content.lower():
        output.append(f"\n--- TABLE CONTENT ---")
        output.append(table_content[:500] + "..." if len(table_content) > 500 else table_content)
        log_table = f"{'='*80}\nPAGE {page_num} - TABLE CONTENT\n{'='*80}\n{table_content}\n\n"
    else:
        output.append(f"\n--- TABLE CONTENT ---")
        output.append("No tables found on this page.")
        log_table = f"{'='*80}\nPAGE {page_num} - TABLE CONTENT\n{'='*80}\nNo tables found on this page.\n\n"
    
    # Prepare visual content
    if visual_content and visual_content.strip() and "no visual elements found" not in visual_content.lower():
        output.append(f"\n--- VISUAL CONTENT ---")
        output.append(visual_content[:500] + "..." if len(visual_content) > 500 else visual_content)
        log_visual = f"{'='*80}\nPAGE {page_num} - VISUAL CONTENT\n{'='*80}\n{visual_content}\n\n"
    else:
        output.append(f"\n--- VISUAL CONTENT ---")
        output.append("No visual elements found on this page.")
        log_visual = f"{'='*80}\nPAGE {page_num} - VISUAL CONTENT\n{'='*80}\nNo visual elements found on this page.\n\n"
    
    output.append(f"{'='*80}\n")
    
    # Thread-safe printing and logging
    if lock:
        # For parallel processing, use tqdm.write for thread-safe output
        from tqdm import tqdm
        tqdm.write('\n'.join(output))
        if log_file:
            log_file.write(log_text)
            log_file.write(log_table)
            log_file.write(log_visual)
    else:
        print('\n'.join(output))
        if log_file:
            log_file.write(log_text)
            log_file.write(log_table)
            log_file.write(log_visual)

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
        
        # Ask for parallel processing
        use_parallel = input("Use parallel processing for faster extraction? (y/n, default: y): ").strip().lower()
        use_parallel = use_parallel != 'n'
        
        try:
            print(f"\nProcessing document: {file_path}")
            
            # Create log file path
            base_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(base_dir, "extraction_logs")
            os.makedirs(output_dir, exist_ok=True)
            
            original_filename = os.path.splitext(os.path.basename(file_path))[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"{original_filename}_{timestamp}.txt"
            log_path = os.path.join(output_dir, log_filename)
            
            # Open log file
            with open(log_path, 'w', encoding='utf-8') as log_file:
                log_file.write(f"{'='*80}\n")
                log_file.write(f"EXTRACTION LOG [SUCCESS]\n")
                log_file.write(f"{'='*80}\n")
                log_file.write(f"Source File: {file_path}\n")
                log_file.write(f"Extraction Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                log_file.write(f"Processing Mode: {'Parallel' if use_parallel else 'Sequential'}\n")
                log_file.write(f"{'='*80}\n\n")
                
                # Ingest document with page-by-page logging
                if use_parallel:
                    extracted_content = rag.ingest_document_with_parallel_logging(
                        file_path, log_file, display_and_log_page_content
                    )
                else:
                    extracted_content = rag.ingest_document_with_logging(
                        file_path, log_file, display_and_log_page_content
                    )
            
            print("\n✓ Document ingested successfully!")
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
        
        # Ask for parallel processing
        use_parallel = input("Use parallel processing for faster extraction? (y/n, default: y): ").strip().lower()
        use_parallel = use_parallel != 'n'
        
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
            
            for idx, filename in enumerate(files, 1):
                file_path = os.path.join(folder_path, filename)
                try:
                    print(f"\n{'#'*80}")
                    print(f"Processing file {idx}/{len(files)}: {filename}...")
                    print(f"{'#'*80}")
                    
                    # Create log file path
                    base_dir = os.path.dirname(os.path.abspath(__file__))
                    output_dir = os.path.join(base_dir, "extraction_logs")
                    os.makedirs(output_dir, exist_ok=True)
                    
                    original_filename = os.path.splitext(os.path.basename(file_path))[0]
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    log_filename = f"{original_filename}_{timestamp}.txt"
                    log_path = os.path.join(output_dir, log_filename)
                    
                    # Open log file
                    with open(log_path, 'w', encoding='utf-8') as log_file:
                        log_file.write(f"{'='*80}\n")
                        log_file.write(f"EXTRACTION LOG [SUCCESS]\n")
                        log_file.write(f"{'='*80}\n")
                        log_file.write(f"Source File: {file_path}\n")
                        log_file.write(f"Extraction Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        log_file.write(f"Processing Mode: {'Parallel' if use_parallel else 'Sequential'}\n")
                        log_file.write(f"{'='*80}\n\n")
                        
                        # Ingest document with page-by-page logging
                        if use_parallel:
                            extracted_content = rag.ingest_document_with_parallel_logging(
                                file_path, log_file, display_and_log_page_content
                            )
                        else:
                            extracted_content = rag.ingest_document_with_logging(
                                file_path, log_file, display_and_log_page_content
                            )
                    
                    print(f"\n✓ Successfully ingested: {filename}")
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
            
            print(f"\n{'='*80}")
            print(f"=== Ingestion Complete ===")
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
            print(f"{'='*80}")
            
        except Exception as e:
            print(f"Error processing folder: {e}")

def inference_mode(rag, session_manager: SessionManager):
    """Handle question answering with mode selection and session support"""
    print("\n=== INFERENCE MODE ===")
    
    # Session management
    print("\nSession Options:")
    print("1. Start new conversation")
    print("2. Continue existing conversation")
    print("3. View session history")
    print("4. Back to Main Menu")
    
    session_choice = input("\nEnter your choice (1-4): ").strip()
    
    if session_choice == "4":
        return
    
    session = None
    
    if session_choice == "1":
        # Create new session
        user_id = input("Enter user ID (default: 'default'): ").strip() or "default"
        session = session_manager.create_session(user_id)
        print(f"✓ New session created: {session.session_id}")
    
    elif session_choice == "2":
        # Continue existing session
        user_id = input("Enter user ID (default: 'default'): ").strip() or "default"
        sessions = session_manager.list_user_sessions(user_id)
        
        if not sessions:
            print("No existing sessions found. Creating new session...")
            session = session_manager.create_session(user_id)
        else:
            print(f"\nFound {len(sessions)} session(s):")
            for idx, s in enumerate(sessions[:5], 1):
                print(f"{idx}. Session {s['session_id'][:8]}... ({s['message_count']} messages) - {s['created_at']}")
            
            choice = input("\nSelect session number (or press Enter for new): ").strip()
            if choice and choice.isdigit() and 1 <= int(choice) <= len(sessions):
                session_id = sessions[int(choice) - 1]['session_id']
                session = session_manager.get_session(session_id)
                print(f"✓ Loaded session: {session_id}")
                
                # Show recent history
                if len(session.history) > 0:
                    print("\n--- Recent Conversation ---")
                    recent = session.get_history(limit=4)
                    for msg in recent:
                        role = "You" if msg["role"] == "user" else "Assistant"
                        content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
                        print(f"{role}: {content}")
                    print("----------------------------\n")
            else:
                session = session_manager.create_session(user_id)
                print(f"✓ New session created: {session.session_id}")
    
    elif session_choice == "3":
        # View session history
        user_id = input("Enter user ID (default: 'default'): ").strip() or "default"
        session_id = input("Enter session ID: ").strip()
        
        session = session_manager.get_session(session_id)
        if not session:
            print("Session not found.")
            return
        
        print(f"\n=== Session History: {session_id} ===")
        print(f"User: {session.user_id}")
        print(f"Created: {session.created_at}")
        print(f"Messages: {len(session.history)}\n")
        
        for idx, msg in enumerate(session.history, 1):
            role = "YOU" if msg["role"] == "user" else "ASSISTANT"
            print(f"\n[{idx}] {role} ({msg['timestamp']})")
            print(f"{msg['content']}\n")
            print("-" * 80)
        
        input("\nPress Enter to continue...")
        return
    
    # Main Q&A loop
    while True:
        print("\n" + "="*50)
        print("Select inference type:")
        print("1. Auto (intelligent selection)")
        print("2. Text-only (faster)")
        print("3. Multimodal (comprehensive)")
        print("4. Clear conversation history")
        print("5. Save and exit to main menu")
        
        mode_choice = input("\nEnter your choice (1-5): ").strip()
        
        if mode_choice == "4":
            if session:
                session.clear_history()
                print("✓ Conversation history cleared.")
            continue
        
        if mode_choice == "5":
            if session:
                session_manager.save_session(session)
                print(f"✓ Session saved: {session.session_id}")
            return
        
        mode_map = {"1": "auto", "2": "text", "3": "multimodal"}
        inference_type = mode_map.get(mode_choice)
        
        if not inference_type:
            print("Invalid choice.")
            continue
        
        print(f"\nInference mode: {inference_type.upper()}")
        print("(Type 'back' to return to menu, 'exit' to quit)")
        
        question = input("Enter your question: ").strip()
        
        if not question:
            print("Please enter a valid question.")
            continue
        
        if question.lower() == 'back':
            break
        
        if question.lower() == 'exit':
            if session:
                session_manager.save_session(session)
                print(f"✓ Session saved: {session.session_id}")
            return
        
        print("\nProcessing your question...")
        try:
            result = rag.answer_question(question, mode=inference_type, session=session)
            
            print(f"\n{'='*80}")
            print(f"ANSWER")
            print(f"{'='*80}")
            print(result["answer"])
            
            print(f"\n{'='*80}")
            print(f"SOURCES")
            print(f"{'='*80}")
            for source in result["sources"]:
                print(f"- {source['source']} (Page {source['page']}, Type: {source['type']})")
            
            print(f"\n{'='*80}")
            print(f"RETRIEVAL INFO")
            print(f"{'='*80}")
            print(f"Documents retrieved: {result['num_documents_retrieved']}")
            
            # Auto-save session after each interaction
            if session:
                session_manager.save_session(session)
            
        except Exception as e:
            print(f"Error answering question: {e}")
            import traceback
            traceback.print_exc()

def main():
    # Initialize the RAG pipeline and session manager
    rag = MultiModalRAGPipeline()
    session_manager = SessionManager()
    
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
        print("3. Session Management")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            ingest_mode(rag)
        
        elif choice == "2":
            inference_mode(rag, session_manager)
        
        elif choice == "3":
            session_management_menu(session_manager)
        
        elif choice == "4":
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")

def session_management_menu(session_manager: SessionManager):
    """Session management submenu"""
    while True:
        print("\n=== SESSION MANAGEMENT ===")
        print("1. List all sessions")
        print("2. Delete session")
        print("3. Cleanup old sessions")
        print("4. Back to Main Menu")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            user_id = input("Enter user ID (default: 'default'): ").strip() or "default"
            sessions = session_manager.list_user_sessions(user_id)
            
            if not sessions:
                print("No sessions found.")
            else:
                print(f"\nFound {len(sessions)} session(s):")
                for idx, s in enumerate(sessions, 1):
                    print(f"{idx}. {s['session_id']} - {s['message_count']} messages - {s['created_at']}")
        
        elif choice == "2":
            session_id = input("Enter session ID to delete: ").strip()
            confirm = input(f"Delete session {session_id}? (y/n): ").strip().lower()
            if confirm == 'y':
                session_manager.delete_session(session_id)
                print("✓ Session deleted.")
        
        elif choice == "3":
            days = input("Delete sessions older than how many days? (default: 30): ").strip()
            days = int(days) if days.isdigit() else 30
            session_manager.cleanup_old_sessions(days)
            print(f"✓ Cleaned up sessions older than {days} days.")
        
        elif choice == "4":
            break

if __name__ == "__main__":
    main()
