# Chat_with_PDFs-Huggingface-Streamlit-

## RAG-PDF Chat Application
A Retrieval-Augmented Generation (RAG) app for chatting with content from uploaded PDFs. Built using Streamlit (frontend), FAISS (vector store), Langchain (conversation chains), and local models for word embeddings. Hugging Face API powers the LLM, supporting natural language queries to retrieve relevant PDF information.

This project is a Retrieval-Augmented Generation (RAG) application that allows users to upload and interact with multiple PDFs by asking natural language questions. The application leverages modern AI techniques for document retrieval and conversational AI to provide contextually accurate answers based on PDF content.

Features
PDF Upload: Users can upload multiple PDFs.
Conversational Query: Ask natural language questions about the uploaded content.
Embedding Models: Uses local word embeddings for efficient text representation.
Vector Store: FAISS is used for storing and retrieving vectors.
LLM Backend: Powered by Hugging Face's free API (FLAN-T5-XXL) for question-answering.
Streamlit UI: Clean and simple user interface using Streamlit.
Langchain: Utilized for creating conversation chains and managing dialogue flow.
Tech Stack
Frontend: Streamlit
Backend: Local embedding models, Hugging Face API (FLAN-T5-XXL)
Vector Store: FAISS
Conversation Management: Langchain
Programming Language: Python
Installation


bash
#### Step 1:Clone the repository:
```
git clone https://github.com/aman167/Chat_with_PDFs-Huggingface-Streamlit-.git

cd Chat_with_PDFs-Huggingface-Streamlit
```
#### Step 2: Step a Virtual Environment
```
conda create -n venv
```
#### Step 3:Activate virtual eviroment
```
conda activate venv/
```
#### Step 4: Install the Requirements 
```
pip install -r requirements.txt
```
#### Step 5: Run the Streamlit app:
```
streamlit run app.py
```

This will open the app in your browser.
Go to the Upload your PDFs boutton seelect PDF/PDFs and upload and click 
#### "PROCESS"
Once the processing is completed start asking questions about the PDF content, and the app will retrieve relevant information.

#### Future Enhancements
 - Support for more complex document types.
 - Integration with cloud-based vector storage.
 - Enhanced conversational flow with more powerfull LLMs.
License
--

Contributions
Feel free to contribute by opening issues or submitting pull requests.
