import streamlit as st
import json
import os
from config import Config
from search_engine import HybridSearchEngine
from document_processor import DocumentProcessor
from logging_config import setup_all_loggers
from typing import Optional
from bulk_processor import BulkProcessor
import tempfile
from pathlib import Path
import pandas as pd
import io

# Initialize logging
loggers = setup_all_loggers()
app_logger = loggers['app']

# Initialize configuration
config = Config()

# Initialize search engine
@st.cache_resource
def get_search_engine():
    app_logger.info("Initializing search engine")
    return HybridSearchEngine(config)

# Initialize document processor
@st.cache_resource
def get_document_processor():
    app_logger.info("Initializing document processor")
    return DocumentProcessor()

def get_bulk_processor() -> Optional[BulkProcessor]:
    """Get or create a bulk processor instance"""
    try:
        app_logger.info("Creating new bulk processor instance")
        if not st.session_state.search_engine:
            raise ValueError("Search engine must be initialized before bulk processor")
        processor = BulkProcessor(st.session_state.search_engine)
        app_logger.info("Bulk processor created successfully")
        return processor
    except Exception as e:
        app_logger.error(f"Failed to create bulk processor: {str(e)}")
        st.error(f"Failed to create bulk processor: {str(e)}")
        return None

def display_bulk_processing_page():
    """Display the bulk question processing page"""
    st.title("🔍 Bulk Question Processing")
    st.markdown("---")
    
    # Information section
    with st.expander("ℹ️ What is Bulk Question Processing?", expanded=False):
        st.markdown("""
        **Purpose:** Get answers to multiple questions from your existing knowledge base
        
        **How it works:**
        1. Upload a file with questions
        2. The system searches your indexed documents
        3. AI generates answers with confidence scores
        4. Download results with answers and source documents
        
        **Note:** This searches existing documents. To add new documents to your knowledge base, use:
        - "CSV Document Upload" for Q&A pairs
        - "JSON Document Upload" for structured documents
        """)
    
    # Format requirements
    with st.expander("📋 File Format Requirements", expanded=False):
        st.markdown("""
        **Required:**
        - CSV or Excel file (.csv or .xlsx)
        - Must contain a `question` column
        - Maximum 1,000 questions per file
        
        **Example CSV Structure:**
        ```csv
        question
        "What is our data security policy?"
        "How do we handle customer complaints?"
        "What are our payment terms?"
        ```
        
        **Notes:**
        - Only the `question` column is used
        - Other columns will be ignored
        - Questions will be processed in parallel for speed
        """)
    
    # Create sample template
    st.subheader("📥 Download Question Template")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if st.button("📥 Download Template", type="secondary"):
            template_data = {
                'question': [
                    'What is our company data security approach?',
                    'How do we handle customer support requests?',
                    'What are our standard payment terms?',
                    'What is our refund policy?',
                    'How do we ensure project quality?'
                ]
            }
            template_df = pd.DataFrame(template_data)
            template_csv = template_df.to_csv(index=False)
            
            st.download_button(
                label="⬇️ Download Questions Template",
                data=template_csv,
                file_name="questions_template.csv",
                mime="text/csv",
                help="Download a sample CSV file with the correct format"
            )
    
    with col2:
        st.info("💡 Download the template to see the required format for bulk questions")
    
    st.markdown("---")
    
    # File upload
    st.subheader("📂 Upload Questions File")
    uploaded_file = st.file_uploader(
        "Choose CSV or Excel file with questions",
        type=['csv', 'xlsx'],
        key="bulk_questions_uploader",
        help="Upload a CSV or Excel file containing questions to get answers from your knowledge base"
    )
    
    if uploaded_file:
        try:
            # File preview and validation
            st.subheader("📋 File Preview & Validation")
            
            # Read file for preview
            if Path(uploaded_file.name).suffix.lower() == '.csv':
                file_contents = uploaded_file.getvalue()
                df = pd.read_csv(io.StringIO(file_contents.decode('utf-8')))
            else:
                df = pd.read_excel(uploaded_file)
            
            # Display file metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Questions", len(df))
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                # Validate required column
                has_question_col = 'question' in df.columns
                validation_status = "✅ Valid" if has_question_col else "❌ Invalid"
                st.metric("Format", validation_status)
            
            # Show column validation
            if not has_question_col:
                st.error("❌ Missing required 'question' column")
                st.info(f"Found columns: {list(df.columns)}")
                st.info("Please ensure your file has a 'question' column containing the questions to process")
                return
            
            # Check question count limit
            if len(df) > 1000:
                st.error(f"❌ Too many questions: {len(df)}. Maximum allowed: 1,000")
                st.info("Please reduce the number of questions in your file")
                return
            
            # Check for empty questions
            valid_questions = df['question'].dropna().astype(str).str.strip()
            valid_questions = valid_questions[valid_questions != '']
            empty_count = len(df) - len(valid_questions)
            
            if empty_count > 0:
                st.warning(f"⚠️ Found {empty_count} empty question(s) that will be skipped")
            
            if len(valid_questions) == 0:
                st.error("❌ No valid questions found in the file")
                st.info("Please ensure your questions are not empty")
                return
            
            st.success(f"✅ File validation passed! {len(valid_questions)} valid questions found")
            
            # Show preview of questions
            st.dataframe(df.head(10), use_container_width=True)
            if len(df) > 10:
                st.info(f"Showing first 10 of {len(df)} questions")
            
            # Process file with bulk processor for final validation
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                # Final validation using bulk processor
                validation_result = st.session_state.bulk_processor.validate_input_file(tmp_file_path)
                
                if not validation_result['is_valid']:
                    st.error(f"❌ Validation failed: {validation_result['error']}")
                    return
                
                st.markdown("---")
                
                # Processing options
                st.subheader("⚙️ Processing Options")
                col1, col2 = st.columns(2)
                with col1:
                    max_workers = st.slider(
                        "Parallel Workers",
                        min_value=1,
                        max_value=10,
                        value=4,
                        help="Number of questions to process simultaneously. More workers = faster processing but higher resource usage."
                    )
                with col2:
                    output_format = st.selectbox(
                        "Output Format",
                        options=['CSV', 'Excel'],
                        help="Format for the results file download"
                    )
                
                # Estimation
                estimated_time = len(valid_questions) / max_workers * 3  # Rough estimate: 3 seconds per question
                st.info(f"📊 Estimated processing time: ~{estimated_time:.1f} seconds for {len(valid_questions)} questions")
                
                if st.button("🚀 Process Questions", type="primary"):
                    with st.spinner(f"Processing {len(valid_questions)} questions..."):
                        try:
                            # Process questions
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            status_text.text("Starting question processing...")
                            results = st.session_state.bulk_processor.process_questions(
                                validation_result['questions'],
                                max_workers=max_workers
                            )
                            progress_bar.progress(100)
                            status_text.text("Processing complete!")
                            
                            # Display results summary
                            st.subheader("📊 Processing Results")
                            success_count = len([r for r in results if r['status'] == 'success'])
                            error_count = len([r for r in results if r['status'] == 'error'])
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("✅ Successful", success_count)
                            with col2:
                                st.metric("❌ Failed", error_count)
                            with col3:
                                success_rate = (success_count / len(results) * 100) if results else 0
                                st.metric("Success Rate", f"{success_rate:.1f}%")
                            
                            # Show sample results if any succeeded
                            if success_count > 0:
                                st.subheader("📝 Sample Results")
                                successful_results = [r for r in results if r['status'] == 'success']
                                sample_result = successful_results[0]
                                
                                with st.expander("View Sample Question & Answer", expanded=False):
                                    st.markdown(f"**Question:** {sample_result['question']}")
                                    st.markdown(f"**Answer:** {sample_result['answer']}")
                                    st.markdown(f"**Confidence:** {sample_result['confidence']:.2f}")
                                    if 'source_documents' in sample_result and sample_result['source_documents']:
                                        st.markdown(f"**Sources:** {len(sample_result['source_documents'])} document(s)")
                            
                            # Export and download results
                            st.subheader("📥 Download Results")
                            export_result = st.session_state.bulk_processor.export_results(
                                results,
                                format=output_format.lower()
                            )
                            
                            col1, col2 = st.columns([1, 2])
                            with col1:
                                st.download_button(
                                    label=f"📁 Download {output_format} Results",
                                    data=export_result['file_data'],
                                    file_name=export_result['file_name'],
                                    mime=export_result['mime_type'],
                                    type="primary"
                                )
                            with col2:
                                st.info(f"💡 Results include questions, answers, confidence scores, and source documents")
                            
                        except Exception as e:
                            st.error(f"❌ Error processing questions: {str(e)}")
                            app_logger.error(f"Error in bulk processing: {str(e)}", exc_info=True)
            
            finally:
                # Clean up temporary file
                try:
                    if 'tmp_file_path' in locals():
                        Path(tmp_file_path).unlink()
                except Exception as e:
                    app_logger.warning(f"Failed to delete temporary file: {str(e)}")
                    
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            st.info("Please ensure the file is a valid CSV or Excel format")
            app_logger.error(f"Error in bulk processing page: {str(e)}", exc_info=True)

def initialize_components():
    """Initialize all required components"""
    try:
        # Initialize config first
        if not st.session_state.config:
            app_logger.info("Initializing configuration")
            st.session_state.config = Config()
            app_logger.info("Configuration initialized successfully")
        
        # Initialize search engine
        if not st.session_state.search_engine:
            app_logger.info("Initializing search engine")
            st.session_state.search_engine = get_search_engine()
            if not st.session_state.search_engine:
                raise ValueError("Failed to initialize search engine")
            app_logger.info("Search engine initialized successfully")
        
        # Initialize document processor
        if not st.session_state.document_processor:
            app_logger.info("Initializing document processor")
            st.session_state.document_processor = get_document_processor()
            if not st.session_state.document_processor:
                raise ValueError("Failed to initialize document processor")
            app_logger.info("Document processor initialized successfully")
        
        # Initialize bulk processor
        if not st.session_state.bulk_processor:
            app_logger.info("Initializing bulk processor")
            st.session_state.bulk_processor = get_bulk_processor()
            if not st.session_state.bulk_processor:
                raise ValueError("Failed to initialize bulk processor")
            app_logger.info("Bulk processor initialized successfully")
        
        app_logger.info("All components initialized successfully")
    except Exception as e:
        app_logger.error(f"Error initializing components: {str(e)}")
        st.error(f"Error initializing components: {str(e)}")
        st.stop()

def display_search_page():
    """Display the search page"""
    st.title("Hybrid Search RFP Assistant")
    
    # Create a form for the search input
    with st.form(key='search_form'):
        # Search input
        query = st.text_input("Enter your question:", key="search_query")
        col1, col2 = st.columns([3, 1])
        with col1:
            top_k = st.slider("Number of results", min_value=1, max_value=10, value=5)
        with col2:
            search_button = st.form_submit_button("Search", type="primary")
    
    # Handle both form submission and button click
    if (search_button or query) and query:  # Check if query is not empty
        app_logger.info(f"Performing search for query: {query}")
        with st.spinner("Searching..."):
            try:
                results = st.session_state.search_engine.search_and_answer(query, top_k)
                
                # Display answer
                st.markdown("### Answer")
                st.markdown(results["answer"])
                
                # Display confidence
                st.progress(results["confidence"], text=f"Confidence: {results['confidence']:.2f}")
                
                # Display search results
                st.markdown("### Search Results")
                for i, result in enumerate(results["search_results"]):
                    with st.expander(f"Result {i+1}: {result['payload']['question'][:100]}..."):
                        st.markdown(f"**Question:** {result['payload']['question']}")
                        st.markdown(f"**Answer:** {result['payload']['answer']}")
                        if "summary" in result["payload"]:
                            st.markdown(f"**Summary:** {result['payload']['summary']}")
                        st.markdown(f"**Relevance Score:** {result['score']:.4f}")
                
                app_logger.info("Search completed successfully")
            except Exception as e:
                app_logger.error(f"Error during search: {str(e)}")
                st.error("An error occurred during the search. Please try again.")

def display_document_upload_page():
    """Display the JSON document upload page"""
    st.title("📄 JSON Document Upload")
    st.markdown("---")
    
    # Information section
    with st.expander("ℹ️ JSON Format Information", expanded=False):
        st.markdown("""
        **Required JSON Structure:**
        ```json
        {
          "documents": [
            {
              "question": "What is our data security approach?",
              "answer": "We implement multi-layered security...",
              "summary": "Security overview",
              "answer_type": "security", 
              "date": "2024-01-01"
            }
          ]
        }
        ```
        
        **Required Fields:** `question`, `answer`  
        **Optional Fields:** `summary`, `answer_type`, `date`
        """)
    
    # File upload section
    st.subheader("📂 Upload JSON File")
    uploaded_file = st.file_uploader(
        "Choose JSON file to upload",
        type=['json'],
        help="Upload a JSON file with the required document structure for indexing"
    )
    
    if uploaded_file is not None:
        try:
            # File preview
            st.subheader("👀 File Preview")
            
            # Read and parse JSON
            file_contents = uploaded_file.getvalue()
            json_data = json.loads(file_contents.decode('utf-8'))
            
            # Validate structure
            if "documents" not in json_data:
                st.error("❌ Invalid JSON structure: Missing 'documents' array")
                st.info("Expected structure: {\"documents\": [...]}")
            else:
                documents = json_data["documents"]
                
                # Display basic info
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Documents", len(documents))
                with col2:
                    st.metric("Format", "✅ Valid JSON")
                
                # Show preview of first few documents
                if documents:
                    st.json(documents[0] if len(documents) == 1 else documents[:3])
                    if len(documents) > 3:
                        st.info(f"Showing first 3 of {len(documents)} documents")
                
                # Index button
                if st.button("🚀 Index JSON Documents", type="primary"):
                    with st.spinner("Processing and indexing JSON documents..."):
                        try:
                            # Save uploaded file to temporary location
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.json', mode='w') as tmp_file:
                                json.dump(json_data, tmp_file)
                                tmp_file_path = tmp_file.name
                            
                            # Process JSON file using existing method
                            app_logger.info(f"Starting JSON document processing from uploaded file")
                            processed_docs = st.session_state.document_processor.process_json_file(tmp_file_path)
                            
                            if processed_docs:
                                # Index the documents
                                indexed_count = st.session_state.search_engine.bulk_index_documents(processed_docs)
                                
                                st.success(f"🎉 Successfully indexed {indexed_count} documents!")
                                
                                # Results summary
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Documents Indexed", indexed_count)
                                with col2:
                                    st.metric("Total Processed", len(processed_docs))
                                
                                app_logger.info(f"Successfully indexed {indexed_count} documents from JSON")
                            else:
                                st.error("❌ No valid documents found in JSON file")
                                st.info("Please check that documents have required 'question' and 'answer' fields")
                            
                            # Clean up temporary file
                            try:
                                Path(tmp_file_path).unlink()
                            except Exception as cleanup_error:
                                app_logger.warning(f"Failed to delete temporary file: {cleanup_error}")
                                
                        except Exception as e:
                            app_logger.error(f"Error processing JSON upload: {str(e)}", exc_info=True)
                            st.error(f"Error processing file: {str(e)}")
                            
                            # Clean up on error
                            try:
                                if 'tmp_file_path' in locals():
                                    Path(tmp_file_path).unlink()
                            except:
                                pass
                            
        except json.JSONDecodeError as e:
            st.error(f"❌ Invalid JSON format: {str(e)}")
            st.info("Please ensure the file is valid JSON with proper syntax")
        except Exception as e:
            app_logger.error(f"Error reading JSON file: {str(e)}")
            st.error(f"Error reading file: {str(e)}")

def display_settings_page():
    """Display the settings page"""
    st.title("Settings")
    
    # Search settings
    st.header("Search Settings")
    current_top_k = st.session_state.config.search_top_k
    new_top_k = st.slider(
        "Number of Search Results",
        min_value=1,
        max_value=20,
        value=current_top_k,
        help="Number of most relevant documents to return for each search"
    )
    
    if new_top_k != current_top_k:
        if st.button("Save Search Settings"):
            try:
                # Update config
                st.session_state.config.search_top_k = new_top_k
                app_logger.info(f"Updated search_top_k to {new_top_k}")
                
                # Update environment variable
                os.environ['SEARCH_TOP_K'] = str(new_top_k)
                
                st.success("Search settings updated successfully!")
            except Exception as e:
                app_logger.error(f"Error updating search settings: {str(e)}", exc_info=True)
                st.error(f"Error updating settings: {str(e)}")

def create_csv_template() -> str:
    """Create a CSV template for users to download - STANDALONE UTILITY FUNCTION"""
    template_data = {
        'question': [
            'What is our company data security approach?',
            'How do we handle customer privacy?',
            'What are our standard payment terms?'
        ],
        'answer': [
            'We implement a multi-layered security approach with encryption at rest and in transit, regular security audits, and strict access controls.',
            'We follow GDPR compliance standards, implement data minimization principles, and provide customers full control over their personal data.',
            'Our standard payment terms are Net 30 days from invoice date, with early payment discounts available for payments within 10 days.'
        ],
        'summary': [
            'Company security overview',
            'Privacy policy summary', 
            'Payment terms explanation'
        ],
        'answer_type': [
            'security',
            'privacy',
            'finance'
        ],
        'date': [
            '2024-01-01',
            '2024-01-01',
            '2024-01-01'
        ]
    }
    
    df = pd.DataFrame(template_data)
    return df.to_csv(index=False)

def display_csv_document_upload_page():
    """NEW FUNCTION: CSV document upload interface - COMPLETELY SEPARATE FROM EXISTING UPLOAD"""
    app_logger.info("Displaying CSV document upload page")
    
    st.title("📊 CSV Document Upload")
    st.markdown("---")
    
    # Information section
    with st.expander("ℹ️ CSV Format Information", expanded=False):
        st.markdown("""
        **Required Columns:**
        - `question`: The question text (required)
        - `answer`: The answer text (required)
        
        **Optional Columns:**
        - `summary`: Brief summary of the Q&A pair
        - `answer_type`: Category/type of the answer (default: 'general')
        - `date`: Date associated with the entry
        - `id`: Custom identifier (UUID generated if not provided)
        
        **Example CSV Structure:**
        ```
        question,answer,summary,answer_type,date
        "What is our security approach?","We use multi-layered security...","Security overview","security","2024-01-01"
        ```
        """)
    
    # Template download section
    st.subheader("📥 Download Template")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if st.button("📥 Download CSV Template", type="secondary"):
            try:
                template_csv = create_csv_template()
                st.download_button(
                    label="⬇️ Download Template File",
                    data=template_csv,
                    file_name="document_template.csv",
                    mime="text/csv",
                    help="Download a sample CSV file with the correct format"
                )
                app_logger.info("CSV template download initiated")
            except Exception as e:
                app_logger.error(f"Error creating CSV template: {str(e)}")
                st.error(f"Error creating template: {str(e)}")
    
    with col2:
        st.info("💡 Download the template to see the required CSV format and example data")
    
    st.markdown("---")
    
    # File upload section
    st.subheader("📂 Upload CSV File")
    uploaded_file = st.file_uploader(
        "Choose CSV file to upload",
        type=['csv'],
        key="csv_document_uploader",  # Different key than existing uploaders
        help="Upload a CSV file with question/answer pairs for indexing into the search database"
    )
    
    if uploaded_file is not None:
        try:
            # File preview
            st.subheader("👀 File Preview")
            
            # Read CSV for preview
            file_contents = uploaded_file.getvalue()
            df_preview = pd.read_csv(io.StringIO(file_contents.decode('utf-8')))
            
            # Display basic info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", len(df_preview))
            with col2:
                st.metric("Columns", len(df_preview.columns))
            with col3:
                required_cols = ['question', 'answer']
                missing_cols = [col for col in required_cols if col not in df_preview.columns]
                validation_status = "✅ Valid" if not missing_cols else "❌ Invalid"
                st.metric("Format", validation_status)
            
            # Show preview of data
            st.dataframe(df_preview.head(10), use_container_width=True)
            
            # Validation feedback
            if missing_cols:
                st.error(f"❌ Missing required columns: {missing_cols}")
                st.info("Required columns: question, answer")
            else:
                st.success("✅ CSV format is valid!")
                
                # Index button
                if st.button("🚀 Index CSV Documents", type="primary"):
                    with st.spinner("Processing and indexing CSV documents..."):
                        try:
                            # Save uploaded file to temporary location
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                                tmp_file.write(file_contents)
                                tmp_file_path = tmp_file.name
                            
                            # Use new CSV indexing method - NO IMPACT ON EXISTING WORKFLOWS
                            app_logger.info(f"Starting CSV indexing from uploaded file")
                            result = st.session_state.search_engine.index_documents_from_csv(tmp_file_path)
                            
                            # Clean up temporary file
                            try:
                                Path(tmp_file_path).unlink()
                            except Exception as cleanup_error:
                                app_logger.warning(f"Failed to delete temporary file: {cleanup_error}")
                            
                            # Display results
                            if result["success"]:
                                st.success(f"🎉 {result['message']}")
                                
                                # Results summary
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Documents Indexed", result["indexed_count"])
                                with col2:
                                    st.metric("Total Processed", result["total_processed"])
                                
                                app_logger.info(f"Successfully indexed {result['indexed_count']} documents from CSV")
                            else:
                                st.error(f"❌ Indexing failed: {result['error']}")
                                if "details" in result:
                                    st.info(f"Details: {result['details']}")
                                app_logger.error(f"CSV indexing failed: {result['error']}")
                                
                        except Exception as e:
                            app_logger.error(f"Error processing CSV upload: {str(e)}", exc_info=True)
                            st.error(f"Error processing file: {str(e)}")
                            
                            # Clean up on error
                            try:
                                if 'tmp_file_path' in locals():
                                    Path(tmp_file_path).unlink()
                            except:
                                pass
                            
        except Exception as e:
            app_logger.error(f"Error reading CSV file: {str(e)}")
            st.error(f"Error reading CSV file: {str(e)}")
            st.info("Please ensure the file is a valid CSV format with proper encoding (UTF-8 recommended)")

def main():
    app_logger.info("Starting application")
    st.set_page_config(
        page_title="Hybrid Search RFP Assistant",
        page_icon="🔍",
        layout="wide"
    )
    
    # Initialize session state variables
    if 'config' not in st.session_state:
        st.session_state.config = None
    if 'search_engine' not in st.session_state:
        st.session_state.search_engine = None
    if 'document_processor' not in st.session_state:
        st.session_state.document_processor = None
    if 'bulk_processor' not in st.session_state:
        st.session_state.bulk_processor = None
    
    # Initialize components
    initialize_components()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Search", "Document Upload", "CSV Document Upload", "Bulk Processing", "Settings"]
    )
    
    # Display selected page
    if page == "Search":
        display_search_page()
    elif page == "Document Upload":
        display_document_upload_page()
    elif page == "CSV Document Upload":
        display_csv_document_upload_page()
    elif page == "Bulk Processing":
        display_bulk_processing_page()
    else:
        display_settings_page()

if __name__ == "__main__":
    main() 