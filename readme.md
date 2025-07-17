# PDF Chat Application

A Streamlit-based application that allows users to chat with their PDF documents using the Groq LLM API and FAISS vector storage. The app processes PDFs, maintains conversation history, and highlights relevant sections in the documents when answering questions.


![PDF Chat App Interface](assets/app_image.png)

Demo video link: https://drive.google.com/file/d/1U4jL301gl0i9bD7BjYX8FRXbbmC-NBiP/view?usp=sharing

Link to test the deployed app: https://pdf-chat-app-joppj72oz2tfnlxsuiqmj5.streamlit.app/

## Features

- 📄 Upload and process multiple PDF documents
- 💬 Chat interface for asking questions about your documents
- 🔍 Smart document search using FAISS vector similarity
- 🎯 Automatic highlighting of relevant sections in PDFs
- 📊 Document processing progress tracking
- 💾 Persistent storage of processed documents
- 📱 Responsive web interface

## Prerequisites

- Python 3.8 or higher
- Groq API key (get one at [groq.com](https://groq.com))

## Installation

1. Clone the repository:
```bash
git clone https://github.com/AayushBhardwaj7/pdf-chat-app
cd pdf-chat-app
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Setup

1. Create a `.env` file in the project root and add your Groq API key:
```
GROQ_API_KEY=your_api_key_here
```

2. If deploying to Streamlit Cloud, add the `GROQ_API_KEY` to your app secrets.

## Running the Application

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser and navigate to `http://localhost:8501`

## Usage

1. **Upload Documents**:
   - Use the file uploader in the left sidebar to select one or more PDF files
   - The app will process each document and display progress
   - Processed documents will be listed below the upload section

2. **Chat with Documents**:
   - Type your questions in the chat input at the bottom
   - The app will search through your documents and provide relevant answers
   - Relevant sections from the source documents will be highlighted
   - Click on document previews to see highlighted sections

3. **View Document Content**:
   - Click on any processed document in the sidebar to preview its content
   - Use the expanders to manage document visibility

## Technical Details

- Uses FAISS for efficient vector similarity search
- Implements sentence-transformers for document embeddings
- Utilizes the Groq LLM API for generating responses
- Stores processed documents and embeddings locally
- Highlights relevant text sections using PyMuPDF

## File Structure

```
pdf-chat-app/
├── app.py              # Main application file
├── requirements.txt    # Python dependencies
├── .env               # Environment variables
├── .gitignore
├── README.md
├── faiss_store/       # FAISS index storage
└── processed_data.pkl # Document tracking data
```

## Requirements

See `requirements.txt` for a complete list of dependencies. Key packages include:

- streamlit
- faiss-cpu
- sentence-transformers
- PyMuPDF
- groq
- langchain

## Deployment

The application can be deployed to Streamlit Cloud:

1. Push your code to GitHub
2. Connect your repository to Streamlit Cloud
3. Add your `GROQ_API_KEY` to the app secrets
4. Deploy!