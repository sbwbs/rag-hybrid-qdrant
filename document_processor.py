import json
import logging
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