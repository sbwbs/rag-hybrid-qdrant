import logging
import pandas as pd
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import json
from datetime import datetime
import uuid
import openpyxl
from search_engine import HybridSearchEngine
from concurrent.futures import ThreadPoolExecutor, as_completed

class BulkProcessor:
    def __init__(self, search_engine: HybridSearchEngine):
        """
        Initialize the BulkProcessor with a search engine.
        
        Args:
            search_engine: Instance of HybridSearchEngine
        """
        self.logger = logging.getLogger('bulk_processor')
        self.logger.propagate = False  # Prevent log propagation
        self.logger.info("Initializing BulkProcessor")
        
        if not search_engine:
            error_msg = "Search engine is required for BulkProcessor initialization"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
            
        self.search_engine = search_engine
        self.logger.info("BulkProcessor initialized successfully")
        
    def validate_input_file(self, file_path: str) -> Dict[str, Any]:
        """
        Validate the input file format and content.
        
        Args:
            file_path: Path to the input file
            
        Returns:
            Dict containing validation results and data if valid
        """
        try:
            self.logger.info(f"Starting file validation for: {file_path}")
            
            # Check if file exists
            if not Path(file_path).exists():
                self.logger.error(f"File does not exist: {file_path}")
                return {
                    'is_valid': False,
                    'error': f"File does not exist: {file_path}"
                }
            
            # Check file extension
            file_ext = Path(file_path).suffix.lower()
            self.logger.info(f"File extension: {file_ext}")
            if file_ext not in ['.csv', '.xlsx']:
                error_msg = f"Unsupported file format: {file_ext}. Only .csv and .xlsx are supported."
                self.logger.error(error_msg)
                return {
                    'is_valid': False,
                    'error': error_msg
                }
            
            # Read file based on extension
            if file_ext == '.csv':
                self.logger.info("Attempting to read CSV file")
                # Read CSV with proper encoding and handle potential issues
                try:
                    df = pd.read_csv(file_path, encoding='utf-8')
                    self.logger.info("Successfully read CSV file with utf-8 encoding")
                except UnicodeDecodeError:
                    self.logger.warning("UTF-8 encoding failed, trying latin1")
                    try:
                        df = pd.read_csv(file_path, encoding='latin1')
                        self.logger.info("Successfully read CSV file with latin1 encoding")
                    except Exception as e:
                        error_msg = f"Failed to read CSV file: {str(e)}"
                        self.logger.error(error_msg, exc_info=True)
                        return {
                            'is_valid': False,
                            'error': error_msg
                        }
            else:
                try:
                    self.logger.info("Reading Excel file")
                    df = pd.read_excel(file_path)
                except Exception as e:
                    error_msg = f"Failed to read Excel file: {str(e)}"
                    self.logger.error(error_msg, exc_info=True)
                    return {
                        'is_valid': False,
                        'error': error_msg
                    }
            
            # Check if DataFrame is empty
            if df.empty:
                error_msg = "File is empty"
                self.logger.error(error_msg)
                return {
                    'is_valid': False,
                    'error': error_msg
                }
            
            self.logger.info(f"File contains {len(df)} rows")
            self.logger.info(f"Columns found: {list(df.columns)}")
            
            # Validate required columns
            required_columns = ['question']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                error_msg = f"Missing required columns: {missing_columns}"
                self.logger.error(error_msg)
                return {
                    'is_valid': False,
                    'error': error_msg
                }
            
            # Clean and validate data
            self.logger.info("Cleaning and validating data")
            try:
                # Convert all questions to strings and handle NaN values
                df['question'] = df['question'].astype(str).str.strip()
                
                # Remove empty questions
                initial_count = len(df)
                df = df[df['question'].str.len() > 0]
                removed_count = initial_count - len(df)
                if removed_count > 0:
                    self.logger.warning(f"Removed {removed_count} empty questions")
                
                if len(df) == 0:
                    error_msg = "No valid questions found after cleaning"
                    self.logger.error(error_msg)
                    return {
                        'is_valid': False,
                        'error': error_msg
                    }
                
                # Check file size limit (1000 questions)
                if len(df) > 1000:
                    error_msg = f"File exceeds limit of 1000 questions. Found: {len(df)}"
                    self.logger.error(error_msg)
                    return {
                        'is_valid': False,
                        'error': error_msg
                    }
                
                # Convert to list of dictionaries with just the question
                self.logger.info("Converting data to question format")
                questions = []
                for idx, row in df.iterrows():
                    try:
                        question_data = {
                            'question': row['question']
                        }
                        questions.append(question_data)
                    except Exception as e:
                        error_msg = f"Error processing row {idx}: {str(e)}"
                        self.logger.error(error_msg, exc_info=True)
                        return {
                            'is_valid': False,
                            'error': error_msg
                        }
                
                self.logger.info(f"Successfully validated file. Found {len(questions)} valid questions")
                return {
                    'is_valid': True,
                    'questions': questions,
                    'total_questions': len(questions)
                }
                
            except Exception as e:
                error_msg = f"Error during data cleaning and validation: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                return {
                    'is_valid': False,
                    'error': error_msg
                }
            
        except Exception as e:
            error_msg = f"Unexpected error validating input file: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {
                'is_valid': False,
                'error': error_msg
            }
    
    def process_questions(self, questions: List[Dict[str, Any]], max_workers: int = 4) -> List[Dict[str, Any]]:
        """
        Process multiple questions in parallel.
        
        Args:
            questions: List of questions to process
            max_workers: Maximum number of parallel workers
            
        Returns:
            List of processed results
        """
        self.logger.info(f"Starting bulk processing of {len(questions)} questions")
        results = []
        
        def process_single_question(question: Dict[str, Any]) -> Dict[str, Any]:
            try:
                self.logger.info(f"Processing question: {question['question']}")
                
                # Step 1: Perform search and get answer using search_and_answer
                result = self.search_engine.search_and_answer(question['question'])
                
                # Step 2: Check if the result indicates an error
                if result.get('status') == 'error':
                    return {
                        "question": question['question'],
                        "answer": None,
                        "confidence": 0.0,
                        "confidence_breakdown": None,
                        "source_documents": [],
                        "status": "error",
                        "error_message": result.get('error_message', 'Unknown error')
                    }
                
                # Step 3: Format source documents
                source_documents = []
                if result.get('search_results'):
                    for doc in result['search_results']:
                        source_documents.append({
                            'content': doc['payload'].get('content', ''),
                            'metadata': doc['payload'],
                            'score': doc['score']
                        })
                
                # Step 4: Return formatted result
                return {
                    "question": question['question'],
                    "answer": result.get("answer"),
                    "confidence": result.get("confidence", 0.0),
                    "confidence_breakdown": result.get("confidence_breakdown"),
                    "source_documents": source_documents,
                    "status": "success"
                }
                
            except Exception as e:
                self.logger.error(f"Error processing question: {str(e)}", exc_info=True)
                return {
                    "question": question['question'],
                    "answer": None,
                    "confidence": 0.0,
                    "confidence_breakdown": None,
                    "source_documents": [],
                    "status": "error",
                    "error_message": str(e)
                }
        
        # Process questions in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_question = {
                executor.submit(process_single_question, question): question 
                for question in questions
            }
            
            for future in as_completed(future_to_question):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Error in parallel processing: {str(e)}", exc_info=True)
        
        # Sort results to match input order
        results.sort(key=lambda x: questions.index(next(q for q in questions if q['question'] == x['question'])))
        
        success_count = len([r for r in results if r['status'] == 'success'])
        self.logger.info(f"Bulk processing completed. Success: {success_count}/{len(questions)}")
        return results
    
    def export_results(self, results: List[Dict[str, Any]], format: str = 'csv') -> Dict[str, Any]:
        """
        Export results to specified format and prepare for download.
        
        Args:
            results: List of processed results
            format: Export format ('csv' or 'excel')
            
        Returns:
            Dict containing:
            - file_path: Path to the exported file
            - file_name: Name of the file
            - file_data: File data for download
        """
        try:
            # Create output directory if it doesn't exist
            output_dir = Path('output')
            output_dir.mkdir(exist_ok=True)
            
            # Generate unique filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"bulk_results_{timestamp}_{uuid.uuid4().hex[:8]}"
            
            # Convert results to DataFrame
            df = pd.DataFrame(results)
            
            # Convert JSON fields to strings
            df['confidence_breakdown'] = df['confidence_breakdown'].apply(
                lambda x: json.dumps(x) if x is not None else None
            )
            df['source_documents'] = df['source_documents'].apply(
                lambda x: json.dumps(x) if x is not None else None
            )
            
            # Export based on format
            if format.lower() == 'csv':
                file_path = output_dir / f"{filename}.csv"
                df.to_csv(file_path, index=False)
                file_data = df.to_csv(index=False).encode('utf-8')
                mime_type = 'text/csv'
            else:
                file_path = output_dir / f"{filename}.xlsx"
                with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name='Results')
                    # Add some basic formatting
                    worksheet = writer.sheets['Results']
                    for column in worksheet.columns:
                        max_length = 0
                        column = [cell for cell in column]
                        for cell in column:
                            try:
                                if len(str(cell.value)) > max_length:
                                    max_length = len(str(cell.value))
                            except:
                                pass
                        adjusted_width = (max_length + 2)
                        worksheet.column_dimensions[column[0].column_letter].width = adjusted_width
                # Read the file for download
                with open(file_path, 'rb') as f:
                    file_data = f.read()
                mime_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            
            self.logger.info(f"Results exported successfully to {file_path}")
            return {
                'file_path': str(file_path),
                'file_name': f"{filename}.{format.lower()}",
                'file_data': file_data,
                'mime_type': mime_type
            }
        except Exception as e:
            self.logger.error(f"Error exporting results: {str(e)}", exc_info=True)
            raise 