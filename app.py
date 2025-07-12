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
    """Display the bulk processing page"""
    st.title("Bulk Question Processing")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Questions (CSV or Excel)",
        type=['csv', 'xlsx'],
        help="Upload a CSV or Excel file containing questions to process"
    )
    
    if uploaded_file:
        # Process file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            # Validate file
            validation_result = st.session_state.bulk_processor.validate_input_file(tmp_file_path)
            
            if not validation_result['is_valid']:
                st.error(f"Invalid file: {validation_result['error']}")
                return
            
            # Display file preview
            st.subheader("File Preview")
            if Path(uploaded_file.name).suffix.lower() == '.csv':
                df = pd.read_csv(tmp_file_path)
            else:
                df = pd.read_excel(tmp_file_path)
            st.dataframe(df.head())
            
            # Processing options
            st.subheader("Processing Options")
            col1, col2 = st.columns(2)
            with col1:
                max_workers = st.slider(
                    "Number of Parallel Workers",
                    min_value=1,
                    max_value=10,
                    value=4,
                    help="Number of questions to process simultaneously"
                )
            with col2:
                output_format = st.selectbox(
                    "Output Format",
                    options=['CSV', 'Excel'],
                    help="Format for the results file"
                )
            
            if st.button("Process Questions", type="primary"):
                with st.spinner("Processing questions..."):
                    try:
                        # Process questions
                        results = st.session_state.bulk_processor.process_questions(
                            validation_result['questions'],
                            max_workers=max_workers
                        )
                        
                        # Display results summary
                        st.subheader("Processing Results")
                        success_count = len([r for r in results if r['status'] == 'success'])
                        error_count = len([r for r in results if r['status'] == 'error'])
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Successfully Processed", success_count)
                        with col2:
                            st.metric("Failed to Process", error_count)
                        
                        # Export and download results
                        st.subheader("Download Results")
                        export_result = st.session_state.bulk_processor.export_results(
                            results,
                            format=output_format.lower()
                        )
                        
                        # Auto-download the file
                        st.download_button(
                            label=f"Download {output_format} File",
                            data=export_result['file_data'],
                            file_name=export_result['file_name'],
                            mime=export_result['mime_type']
                        )
                        
                    except Exception as e:
                        st.error(f"Error processing questions: {str(e)}")
                        app_logger.error(f"Error in bulk processing: {str(e)}", exc_info=True)
        
        finally:
            # Clean up temporary file
            try:
                Path(tmp_file_path).unlink()
            except Exception as e:
                app_logger.warning(f"Failed to delete temporary file: {str(e)}")

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
    """Display the document upload page"""
    st.title("Document Upload")
    
    uploaded_files = st.file_uploader(
        "Upload RFP Documents",
        type=['pdf', 'docx', 'txt'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                try:
                    # Create temporary directory for uploaded files
                    with tempfile.TemporaryDirectory() as temp_dir:
                        temp_path = Path(temp_dir)
                        
                        # Save uploaded files to temporary directory
                        for file in uploaded_files:
                            file_path = temp_path / file.name
                            with open(file_path, 'wb') as f:
                                f.write(file.getbuffer())
                        
                        # Process documents
                        success_count = 0
                        error_count = 0
                        
                        for file_path in temp_path.glob('*'):
                            try:
                                st.session_state.document_processor.process_document(str(file_path))
                                success_count += 1
                            except Exception as e:
                                app_logger.error(f"Error processing {file_path.name}: {str(e)}")
                                error_count += 1
                        
                        if success_count > 0:
                            st.success(f"Successfully processed {success_count} document(s)")
                        if error_count > 0:
                            st.error(f"Failed to process {error_count} document(s)")
                            
                except Exception as e:
                    app_logger.error(f"Error in document processing: {str(e)}")
                    st.error(f"Error processing documents: {str(e)}")

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
        ["Search", "Document Upload", "Bulk Processing", "Settings"]
    )
    
    # Display selected page
    if page == "Search":
        display_search_page()
    elif page == "Document Upload":
        display_document_upload_page()
    elif page == "Bulk Processing":
        display_bulk_processing_page()
    else:
        display_settings_page()

if __name__ == "__main__":
    main() 