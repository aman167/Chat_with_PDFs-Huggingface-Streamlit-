import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from huggingface_hub import snapshot_download
from langchain_community.llms import Ollama
from transformers import AutoModelForCausalLM, AutoTokenizer, LongformerTokenizer, LongformerForQuestionAnswering
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
import requests

# Define your API endpoint and model
#API_URL = "https://api-inference.huggingface.co/models/gpt2"
#headers = {"Authorization": f"Bearer YOUR_API_KEY"}

class HuggingFaceLLM:
    """Wrapper class for Hugging Face Inference API."""
    
    def __call__(self, prompt):
        """Call the API with the prompt."""
        data = {"inputs": prompt}
        response = requests.post(API_URL, headers=headers, json=data)
        
        if response.status_code == 200:
            return response.json()[0]['generated_text']  # Adjust based on the actual response structure
        else:
            raise Exception(f"API call failed: {response.status_code}, {response.text}")


def get_pdf_text(pdf_docs):
    """
    Extracts text from a list of PDF documents.

    Args:
        pdf_docs (list): A list of PDF file objects.

    Returns:
        str: The extracted text from the PDF documents.
    """

    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    """
    Splits a given text into chunks, which will be used to
    generate embeddings and be stored in the vector store.

    Parameters:
    text (str): The text to be split into chunks.

    Returns:
    list of str: A list of text chunks.
    """
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    """
    Creates a vector store from text chunks using embeddings.

    Parameters:
    text_chunks (list of str): A list of text chunks to be converted into embeddings.

    Returns:
    FAISS: A vector store containing the embeddings of the provided text chunks.
    """
    #embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore




def get_conversation_chain(vectorstore):
#------------------------------------------------------------Uncomment if using Ollama or OpenAI----------------------------------------------------
#   #llm = ChatOpenAI()
    #llm = Ollama(model = "llama3.1")
    #llm = HuggingFaceHub(repo_id="meta-llama/Meta-Llama-3-8B", model_kwargs={"temperature":0.3, "max_length":512})
    #memory = ConversationBufferMemory(
    #    memory_key='chat_history', return_messages=True)
    #conversation_chain = ConversationalRetrievalChain.from_llm(
    # Use the HuggingFaceLLM wrapper
    #    llm = HuggingFaceLLM(),
    #    retriever=vectorstore.as_retriever(),
    #    memory=memory
    #)
#-------------------------------------------------------------------------------------------------------------------------------------------------

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)

    # Create a Hugging Face pipeline
    hf_pipeline = pipeline("text-generation", model="openai-community/gpt2", 
                            tokenizer="openai-community/gpt2", max_length=1024)

    # Wrap the pipeline in HuggingFacePipeline for LangChain
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    return conversation_chain


def handle_userinput(user_question):
    """
    Handles user input by passing it to the conversational retrieval chain, updating
    the chat history, and displaying the chat history in the Streamlit app.

    Parameters:
    user_question (str): The user's question or input.

    Returns:
    None
    """
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    """
    Main entry point of the application. Sets up the Streamlit app
    by loading the CSS styles, setting up the page configuration,
    and displaying the main chat interface. It also handles user
    input by passing it to the conversational retrieval chain and
    updating the chat history.

    Parameters:
    None

    Returns:
    None
    """
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",
                        page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)


if __name__ == '__main__':
    main()
