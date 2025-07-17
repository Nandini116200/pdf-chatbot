from dotenv import load_dotenv
load_dotenv()
import os
import json
import torch
import fitz
import base64
import streamlit as st
import tempfile
from typing import List, Dict, Tuple
from datetime import datetime
from dataclasses import dataclass
from PyPDF2 import PdfReader
from groq import Groq
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import hashlib
import pickle

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "./models"

# Data Classes
@dataclass
class TextChunk:
    """Represents a chunk of text from a PDF with location information"""
    text: str
    page_num: int
    bbox: tuple  # (x0, y0, x1, y1)

class PDFChatBot:
    """Main class for handling PDF processing and chat interactions"""
    
    def __init__(self, groq_api_key: str):
        """Initialize the chatbot with necessary components"""
        # Initialize Groq client for LLM
        self.groq_client = Groq(api_key=groq_api_key)
        
        # Setup embeddings model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        st.info(f"Using device: {device} for embeddings")
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name='all-MiniLM-L6-v2',
            model_kwargs={'device': device}
        )
        
        # Initialize or load FAISS index
        self.vector_store = self._load_or_create_vector_store()
        
        # Track processed documents
        self.processed_docs = {}
        self.processed_files = set()
    
    def _load_or_create_vector_store(self):
        """Load existing vector store or create a new one"""
        if os.path.exists("faiss_store"):
            try:
                vector_store = FAISS.load_local("faiss_store", self.embeddings, allow_dangerous_deserialization=True)
                # Load processed docs and files
                if os.path.exists("processed_data.pkl"):
                    with open("processed_data.pkl", "rb") as f:
                        data = pickle.load(f)
                        self.processed_docs = data.get("processed_docs", {})
                        self.processed_files = data.get("processed_files", set())
                return vector_store
            except Exception as e:
                st.warning(f"Error loading existing index: {e}. Creating new one.")
        return FAISS.from_texts([""], self.embeddings)
    
    def _save_vector_store(self):
        """Save vector store and processed documents info"""
        self.vector_store.save_local("faiss_store")
        with open("processed_data.pkl", "wb") as f:
            pickle.dump({
                "processed_docs": self.processed_docs,
                "processed_files": self.processed_files
            }, f)
    
    def extract_text_with_locations(self, pdf_path: str) -> List[TextChunk]:
        """Extract text and its location information from PDF"""
        chunks = []
        doc = fitz.open(pdf_path)
        
        progress_bar = st.progress(0)
        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            chunks.append(TextChunk(
                                text=span["text"],
                                page_num=page_num,
                                bbox=(span["bbox"])
                            ))
            
            progress_bar.progress((page_num + 1) / len(doc))
        
        progress_bar.empty()
        doc.close()
        return chunks

    def process_pdf(self, pdf_file, filename: str, chunk_size: int = 500) -> dict:
        """Process PDF file and store chunks in vector database"""
        # Generate document ID and check for duplicates
        doc_id = hashlib.md5(pdf_file.read()).hexdigest()
        pdf_file.seek(0)
        
        if doc_id in self.processed_docs:
            return {
                "status": "skipped",
                "message": f"Document {filename} was already processed"
            }
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_file.read())
            tmp_path = tmp_file.name
        
        # Extract and process text
        with st.spinner('Extracting text from PDF...'):
            chunks = self.extract_text_with_locations(tmp_path)
            
        with st.spinner('Processing content...'):
            # Combine spans into chunks
            current_chunk = []
            current_text = ""
            chunk_locations = []
            
            for chunk in chunks:
                if not current_chunk:
                    current_chunk.append(chunk)
                    current_text = chunk.text + " "
                elif len(current_text) + len(chunk.text) > chunk_size:
                    chunk_locations.append(current_chunk)
                    current_chunk = [chunk]
                    current_text = chunk.text + " "
                else:
                    current_chunk.append(chunk)
                    current_text += chunk.text + " "
            
            if current_chunk:
                chunk_locations.append(current_chunk)
            
            # Prepare for database storage
            texts = []
            metadatas = []
            
            for i, chunk_group in enumerate(chunk_locations):
                text = " ".join(c.text for c in chunk_group)
                locations = [{
                    "page": c.page_num,
                    "bbox": tuple(float(x) for x in c.bbox)
                } for c in chunk_group]
                
                texts.append(text)
                metadatas.append({
                    "doc_id": doc_id,
                    "source": filename,
                    "chunk_index": i,
                    "total_chunks": len(chunk_locations),
                    "locations_json": json.dumps(locations),
                    "processed_date": datetime.now().isoformat()
                })
            
            # Add to FAISS index
            new_vector_store = FAISS.from_texts(
                texts,
                self.embeddings,
                metadatas=metadatas
            )
            
            if hasattr(self.vector_store, 'index') and self.vector_store.index.ntotal > 0:
                self.vector_store.merge_from(new_vector_store)
            else:
                self.vector_store = new_vector_store
            
            # Update tracking and save
            self.processed_docs[doc_id] = tmp_path
            self.processed_files.add(filename)
            self._save_vector_store()
            
            return {
                "status": "success",
                "message": f"Processed {len(chunks)} text blocks from {filename}",
                "text": " ".join(texts),
                "chunks": len(chunks)
            }

    def highlight_pdf(self, pdf_path: str, locations: List[Dict]) -> bytes:
        """Add yellow highlights around relevant sections in the PDF"""
        doc = fitz.open(pdf_path)
        highlight_color = (1, 0.9, 0)  # Yellow
        
        # Group locations by page
        page_locations = {}
        for loc in locations:
            page_num = loc["page"]
            if page_num not in page_locations:
                page_locations[page_num] = []
            page_locations[page_num].append(loc["bbox"])
        
        # Process each page
        for page_num, bboxes in page_locations.items():
            if page_num >= len(doc):
                continue
                
            page = doc[page_num]
            
            # Merge overlapping or very close rectangles
            merged_bboxes = []
            for bbox in sorted(bboxes, key=lambda x: (x[1], x[0])):  # Sort by y, then x
                if not merged_bboxes:
                    merged_bboxes.append(list(bbox))
                    continue
                    
                last_bbox = merged_bboxes[-1]
                # Check if rectangles overlap or are very close
                if (bbox[0] <= last_bbox[2] + 5 and  # Close horizontally
                    bbox[1] <= last_bbox[3] + 5 and  # Close vertically
                    bbox[2] >= last_bbox[0] - 5):
                    # Merge the rectangles
                    last_bbox[0] = min(last_bbox[0], bbox[0])
                    last_bbox[1] = min(last_bbox[1], bbox[1])
                    last_bbox[2] = max(last_bbox[2], bbox[2])
                    last_bbox[3] = max(last_bbox[3], bbox[3])
                else:
                    merged_bboxes.append(list(bbox))
            
            # Add highlights with padding
            for bbox in merged_bboxes:
                # Add small padding
                padding = 2
                rect = fitz.Rect(
                    bbox[0] - padding,
                    bbox[1] - padding,
                    bbox[2] + padding,
                    bbox[3] + padding
                )
                # Draw rectangle without rounded corners
                page.draw_rect(rect, color=highlight_color, width=1.5)
        
        return doc.write()

    def get_relevant_context(self, query: str, n_results: int = 5) -> Tuple[Dict, List[Dict]]:
        """Get relevant context and locations for the query"""
        try:
            results = self.vector_store.similarity_search_with_score(query, k=n_results + 5)
            
            contexts = {}
            highlight_locations = []
            
            filtered_results = [(doc, score) for doc, score in results]
            
            for doc, score in filtered_results:
                try:
                    source = doc.metadata.get('source', 'Unknown Document')
                    doc_id = doc.metadata.get('doc_id', '')
                    
                    if source not in contexts:
                        contexts[source] = {
                            'content': [],
                            'relevance': 1.0,
                            'doc_id': doc_id
                        }
                    contexts[source]['content'].append(doc.page_content)
                    
                    locations_json = doc.metadata.get('locations_json', '[]')
                    try:
                        locations = json.loads(locations_json)
                        valid_locations = [
                            loc for loc in locations 
                            if isinstance(loc.get('bbox'), (list, tuple)) 
                            and len(loc['bbox']) == 4
                        ]
                        highlight_locations.extend(valid_locations)
                    except (json.JSONDecodeError, KeyError):
                        continue
                        
                except Exception as e:
                    continue
            
            return contexts, highlight_locations
            
        except Exception as e:
            st.error(f"Error in get_relevant_context: {str(e)}")
            return {}, []

    def generate_response(self, query: str, contexts: Dict) -> str:
        """Generate response using Groq's LLM"""
        formatted_contexts = ""
        for source, data in contexts.items():
            formatted_contexts += f"\nFrom {source} (relevance: {data['relevance']:.2f}):\n"
            formatted_contexts += "\n".join(data['content'])
        
        prompt = f"""Context from multiple documents:
{formatted_contexts}

Question: {query}

Please provide a comprehensive response based on the context above. If referring to specific information, mention which document it came from. If the context doesn't contain relevant information, please indicate that."""
        
        with st.spinner('Generating response...'):
            completion = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context from multiple documents. Always cite the source document when providing information."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            return completion.choices[0].message.content

    def chat(self, query: str) -> Tuple[str, Dict[str, bytes]]:
        """Main chat function that handles the complete conversation flow"""
        contexts, locations = self.get_relevant_context(query)
        
        if not contexts:
            return "I couldn't find any relevant information in the uploaded documents.", {}
        
        response = self.generate_response(query, contexts)
        highlighted_pdfs = {}
        
        for source, data in contexts.items():
            doc_id = data['doc_id']
            if doc_id in self.processed_docs:
                pdf_path = self.processed_docs[doc_id]
                source_locations = [loc for loc in locations if loc['page'] < fitz.open(pdf_path).page_count]
                if source_locations:  # Only create highlighted PDF if there are valid locations
                    highlighted_pdf = self.highlight_pdf(pdf_path, source_locations)
                    highlighted_pdfs[source] = highlighted_pdf
        
        return response, highlighted_pdfs

    def get_document_list(self) -> List[str]:
        """Get list of all processed documents"""
        return sorted(list(self.processed_files))

def initialize_session_state():
    """Initialize Streamlit session state"""
    if 'chatbot' not in st.session_state:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            st.error("Please set the GROQ_API_KEY environment variable")
            st.stop()
        st.session_state.chatbot = PDFChatBot(groq_api_key)
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

def main():                                        #streamlit_code
    """Main application function"""
    st.set_page_config(page_title="PDF Chat App", page_icon="📚", layout="wide")
    st.title("📚 PDF ChatBot")
    
    initialize_session_state()
    
    # Create two columns for layout
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.header("Upload PDFs")
        uploaded_files = st.file_uploader(
            "Choose PDF files", 
            type="pdf",
            accept_multiple_files=True,
            help="Upload one or more PDF files to chat with"
        )
        
        if uploaded_files:
            for pdf_file in uploaded_files:
                result = st.session_state.chatbot.process_pdf(pdf_file, pdf_file.name)
                if result["status"] == "success":
                    st.success(result["message"])
                    with st.expander(f"Preview content from {pdf_file.name}", expanded=False):
                        st.text_area(
                            "Document content",
                            result["text"],
                            height=400,
                            disabled=True
                        )
                else:
                    st.info(result["message"])
        
        st.header("Processed Documents")
        documents = st.session_state.chatbot.get_document_list()
        if documents:
            for doc in documents:
                st.write(f"📄 {doc}")
        else:
            st.info("No documents processed yet. Please upload some PDFs.")
    
    with col2:
        st.header("Chat")
        
        # Display chat history
        for role, message, highlighted_pdfs in st.session_state.chat_history:
            with st.chat_message(role):
                st.write(message)
                if role == "assistant" and highlighted_pdfs:
                    st.markdown("---")
                    st.markdown("**Download highlighted PDFs:**")
                    for source, pdf_bytes in highlighted_pdfs.items():
                        st.download_button(
                            label=f"📥 Download highlighted {source}",
                            data=pdf_bytes,
                            file_name=f"highlighted_{source}",
                            mime="application/pdf",
                            key=f"download_{source}_{hash(message)}"  # Unique key for each button
                        )
        
        # Chat input
        if query := st.chat_input("Ask a question about your documents"):
            st.session_state.chat_history.append(("user", query, {}))
            with st.chat_message("user"):
                st.write(query)
            
            try:
                response, highlighted_pdfs = st.session_state.chatbot.chat(query)
                st.session_state.chat_history.append(("assistant", response, highlighted_pdfs))
                
                with st.chat_message("assistant"):
                    st.write(response)
                    if highlighted_pdfs:
                        st.markdown("---")
                        st.markdown("**Download highlighted PDFs:**")
                        for source, pdf_bytes in highlighted_pdfs.items():
                            st.download_button(
                                label=f"📥 Download highlighted {source}",
                                data=pdf_bytes,
                                file_name=f"highlighted_{source}",
                                mime="application/pdf",
                                key=f"download_{source}_{hash(response)}"  # Unique key for each button
                            )
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")

if __name__ == "__main__":
    main()