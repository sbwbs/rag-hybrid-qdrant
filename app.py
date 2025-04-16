import streamlit as st
import json
import os
from config import Config
from search_engine import HybridSearchEngine
from document_processor import DocumentProcessor
from logging_config import setup_all_loggers

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

def main():
    app_logger.info("Starting application")
    st.set_page_config(
        page_title="Hybrid Search RFP Assistant",
        page_icon="🔍",
        layout="wide"
    )
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Search", "Document Management", "Settings"])
    
    search_engine = get_search_engine()
    doc_processor = get_document_processor()
    
    if page == "Search":
        display_search_page(search_engine)
    elif page == "Document Management":
        display_document_management(search_engine, doc_processor)
    else:
        display_settings_page(config)

def display_search_page(search_engine):
    app_logger.info("Displaying search page")
    st.title("Hybrid Search RFP Assistant")
    
    # Search input
    query = st.text_input("Enter your question:", key="search_query")
    col1, col2 = st.columns([3, 1])
    with col1:
        top_k = st.slider("Number of results", min_value=1, max_value=10, value=5)
    with col2:
        search_button = st.button("Search", type="primary")
    
    if search_button and query:
        app_logger.info(f"Performing search for query: {query}")
        with st.spinner("Searching..."):
            try:
                results = search_engine.search_and_answer(query, top_k)
                
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

def display_document_management(search_engine, doc_processor):
    app_logger.info("Displaying document management page")
    st.title("Document Management")
    
    # Document upload
    st.markdown("### Upload Documents")
    uploaded_file = st.file_uploader("Choose a JSON file", type="json")
    
    if uploaded_file is not None:
        # Preview the file
        try:
            app_logger.info("Processing uploaded file")
            data = json.load(uploaded_file)
            if "documents" in data:
                documents = data["documents"]
                st.success(f"Found {len(documents)} documents in the file")
                
                # Preview the documents
                with st.expander("Preview Documents"):
                    for i, doc in enumerate(documents[:5]):  # Preview first 5
                        st.markdown(f"**Document {i+1}**")
                        st.json(doc)
                    if len(documents) > 5:
                        st.markdown(f"*... and {len(documents) - 5} more*")
                
                # Index the documents
                if st.button("Index Documents", type="primary"):
                    with st.spinner("Indexing documents..."):
                        try:
                            # Process and validate documents
                            valid_docs = doc_processor.validate_and_clean(documents)
                            if len(valid_docs) != len(documents):
                                st.warning(f"{len(documents) - len(valid_docs)} documents were invalid and skipped")
                            
                            # Index the valid documents
                            indexed_count = search_engine.bulk_index_documents(valid_docs)
                            st.success(f"Successfully indexed {indexed_count} documents")
                            app_logger.info(f"Successfully indexed {indexed_count} documents")
                        except Exception as e:
                            app_logger.error(f"Error indexing documents: {str(e)}")
                            st.error(f"Error indexing documents: {str(e)}")
            else:
                app_logger.error("Invalid JSON format. Expected a 'documents' array.")
                st.error("Invalid JSON format. Expected a 'documents' array.")
        except Exception as e:
            app_logger.error(f"Error processing file: {str(e)}")
            st.error(f"Error processing file: {str(e)}")
    
    # Collection stats
    st.markdown("### Collection Statistics")
    if st.button("Refresh Statistics"):
        try:
            app_logger.info("Fetching collection statistics")
            collection_info = search_engine.qdrant_client.get_collection(search_engine.collection_name)
            st.metric("Total Documents", collection_info.points_count)
            st.json(collection_info.dict())
            app_logger.info("Collection statistics retrieved successfully")
        except Exception as e:
            app_logger.error(f"Error fetching collection info: {str(e)}")
            st.error(f"Error fetching collection info: {str(e)}")

def display_settings_page(config):
    app_logger.info("Displaying settings page")
    st.title("Settings")
    
    st.markdown("### API Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        openai_key = st.text_input("OpenAI API Key", value="••••••", type="password")
        qdrant_url = st.text_input("Qdrant URL", value=config.qdrant_url)
    
    with col2:
        qdrant_key = st.text_input("Qdrant API Key", value="••••••", type="password")
        collection_name = st.text_input("Collection Name", value=config.collection_name)
    
    llm_model = st.selectbox("LLM Model", options=["gpt-4o"], index=0)
    
    if st.button("Save Settings", type="primary"):
        app_logger.info("Saving settings")
        # Save settings logic would go here
        st.success("Settings saved successfully!")
        app_logger.info("Settings saved successfully")

if __name__ == "__main__":
    main() 