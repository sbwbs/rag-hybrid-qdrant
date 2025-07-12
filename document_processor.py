import json
import logging
import pandas as pd
import uuid
from typing import List, Dict, Any, Optional

# Get logger
logger = logging.getLogger('document_processor')

class DocumentProcessor:
    """Handles document processing and validation for the hybrid search engine"""
    
    def __init__(self):
        logger.info("Initializing DocumentProcessor")
    
    def validate_document(self, document: Dict[str, Any]) -> bool:
        """Validate if a document has the required fields and correct structure"""
        logger.debug("Validating document structure")
        
        try:
            # Check required fields
            required_fields = ["question", "answer"]
            for field in required_fields:
                if field not in document or not document[field]:
                    logger.warning(f"Missing required field: {field}")
                    return False
            
            # Validate field types
            if not isinstance(document["question"], str) or not isinstance(document["answer"], str):
                logger.warning("Question and answer must be strings")
                return False
            
            # Validate optional fields if present
            optional_fields = ["summary", "answer_type", "date"]
            for field in optional_fields:
                if field in document and document[field] is not None:
                    if not isinstance(document[field], str):
                        logger.warning(f"Optional field {field} must be a string")
                        return False
            
            return True
        except Exception as e:
            logger.error(f"Error validating document: {str(e)}")
            return False
    
    def clean_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and normalize document fields"""
        logger.debug("Cleaning document fields")
        
        try:
            # Create a copy to avoid modifying the original
            cleaned = document.copy()
            
            # Clean text fields
            for field in ["question", "answer", "summary"]:
                if field in cleaned and cleaned[field]:
                    cleaned[field] = cleaned[field].strip()
            
            # Ensure optional fields exist with default values
            cleaned.setdefault("summary", "")
            cleaned.setdefault("answer_type", "general")
            cleaned.setdefault("date", "")
            
            return cleaned
        except Exception as e:
            logger.error(f"Error cleaning document: {str(e)}")
            raise
    
    def validate_and_clean(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and clean a list of documents"""
        logger.info(f"Processing {len(documents)} documents")
        
        try:
            valid_documents = []
            for i, doc in enumerate(documents, 1):
                logger.debug(f"Processing document {i}/{len(documents)}")
                if self.validate_document(doc):
                    cleaned_doc = self.clean_document(doc)
                    valid_documents.append(cleaned_doc)
                else:
                    logger.warning(f"Skipping invalid document {i}")
            
            logger.info(f"Successfully processed {len(valid_documents)} documents")
            return valid_documents
        except Exception as e:
            logger.error(f"Error validating and cleaning documents: {str(e)}")
            raise
    
    def process_json_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Process a JSON file and return valid documents"""
        logger.info(f"Processing JSON file: {file_path}")
        
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
            
            if "documents" in data:
                documents = data["documents"]
                return self.validate_and_clean(documents)
            else:
                logger.error("Invalid JSON format. Expected a 'documents' array.")
                raise ValueError("Invalid JSON format. Expected a 'documents' array.")
                
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON file: {str(e)}")
            raise ValueError(f"Invalid JSON file: {str(e)}")
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
        except Exception as e:
            logger.error(f"Unexpected error processing file: {str(e)}")
            raise 
    
    def process_csv_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Process CSV file and convert to document format - ADDITIVE METHOD, NO IMPACT ON EXISTING CODE"""
        logger.info(f"Processing CSV file: {file_path}")
        
        try:
            # Try to read CSV with UTF-8 encoding first, fallback to Latin1
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
                logger.debug("Successfully read CSV with UTF-8 encoding")
            except UnicodeDecodeError:
                logger.debug("UTF-8 failed, trying Latin1 encoding")
                df = pd.read_csv(file_path, encoding='latin1')
                logger.debug("Successfully read CSV with Latin1 encoding")
            
            # Validate required columns exist
            required_cols = ['question', 'answer']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                error_msg = f"Missing required columns: {missing_cols}. Found columns: {list(df.columns)}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            logger.info(f"CSV file has {len(df)} rows and columns: {list(df.columns)}")
            
            # Convert CSV rows to document format - EXACT SAME FORMAT as JSON processing
            documents = []
            for idx, row in df.iterrows():
                try:
                    # Create document in exact same format as JSON processing
                    doc = {
                        'question': str(row['question']).strip() if pd.notna(row['question']) else '',
                        'answer': str(row['answer']).strip() if pd.notna(row['answer']) else '',
                        'summary': str(row.get('summary', '')).strip() if pd.notna(row.get('summary')) else '',
                        'answer_type': str(row.get('answer_type', 'general')).strip() if pd.notna(row.get('answer_type')) else 'general',
                        'date': str(row.get('date', '')).strip() if pd.notna(row.get('date')) else '',
                        'id': str(row.get('id', str(uuid.uuid4()))).strip() if pd.notna(row.get('id')) else str(uuid.uuid4())
                    }
                    
                    # REUSE EXISTING validation logic - no new validation code
                    if self.validate_document(doc):
                        cleaned_doc = self.clean_document(doc)
                        documents.append(cleaned_doc)
                        logger.debug(f"Successfully processed row {idx + 1}")
                    else:
                        logger.warning(f"Invalid document at row {idx + 1} (CSV row {idx + 2}): validation failed")
                        
                except Exception as e:
                    logger.warning(f"Error processing row {idx + 1} (CSV row {idx + 2}): {str(e)}")
                    continue
            
            logger.info(f"Successfully converted {len(documents)} documents from CSV")
            return documents
            
        except pd.errors.EmptyDataError:
            error_msg = "CSV file is empty"
            logger.error(error_msg)
            raise ValueError(error_msg)
        except FileNotFoundError:
            error_msg = f"CSV file not found: {file_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        except Exception as e:
            error_msg = f"Error processing CSV file: {str(e)}"
            logger.error(error_msg)
            raise
    
    def _csv_to_documents(self, csv_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Private helper: Convert CSV DataFrame to document format - HELPER METHOD ONLY"""
        logger.debug("Converting CSV data to document format")
        
        documents = []
        for idx, row in csv_data.iterrows():
            try:
                doc = {
                    'question': str(row['question']).strip() if pd.notna(row['question']) else '',
                    'answer': str(row['answer']).strip() if pd.notna(row['answer']) else '',
                    'summary': str(row.get('summary', '')).strip() if pd.notna(row.get('summary')) else '',
                    'answer_type': str(row.get('answer_type', 'general')).strip() if pd.notna(row.get('answer_type')) else 'general',
                    'date': str(row.get('date', '')).strip() if pd.notna(row.get('date')) else '',
                    'id': str(row.get('id', str(uuid.uuid4()))).strip() if pd.notna(row.get('id')) else str(uuid.uuid4())
                }
                
                # Use existing validation
                if self.validate_document(doc):
                    documents.append(self.clean_document(doc))
                    
            except Exception as e:
                logger.warning(f"Error converting row {idx}: {str(e)}")
                continue
        
        return documents