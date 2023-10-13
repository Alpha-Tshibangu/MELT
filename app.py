import streamlit as st
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders.image import UnstructuredImageLoader
from langchain.document_loaders import ImageCaptionLoader
from langchain.docstore.document import Document
import os

# Chat UI title
st.header("Machine Enabled Legislation Transposition")

# File uploader in the sidebar on the left
with st.sidebar:
    # Input for OpenAI API Key
    openai_api_key = st.text_input("OpenAI API Key", type="password")

    # Check if OpenAI API Key is provided
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    # Set OPENAI_API_KEY as an environment variable
    os.environ["OPENAI_API_KEY"] = openai_api_key

# Initialize ChatOpenAI model
llm = ChatOpenAI(temperature=0, max_tokens=1000, model_name="gpt-4", streaming=True)

# Load usage instructions from the text file
def load_version_history():
    with open("usage_instructions.txt", "r") as file:
        return file.read()

# Sidebar section for uploading files and providing a website URL
with st.sidebar:
    uploaded_files = st.file_uploader("Please upload your files", accept_multiple_files=True, type=None)
    website_url = st.text_input("Website URL")

    # Create an expander for the usage instructions in the sidebar
    with st.sidebar.expander("**Usage Instructions**", expanded=False):
        st.write(load_version_history())

    st.info("Please refresh the browser if you decide to upload more files to reset the session.", icon="🚨")

# Check if files are uploaded or website URL is provided
if uploaded_files or website_url:
    # Print the number of files uploaded or website URL provided to the console
    st.write(f"Number of files uploaded: {len(uploaded_files)}")

    # Load the data and perform preprocessing only if it hasn't been loaded before
    if "processed_data" not in st.session_state:
        # Load the data from uploaded files
        documents = []

        if uploaded_files:
            for uploaded_file in uploaded_files:
                # Get the full file path of the uploaded file
                file_path = os.path.join(os.getcwd(), uploaded_file.name)

                # Save the uploaded file to disk
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                # Check if the file is an image
                if file_path.endswith((".png", ".jpg")):
                    # Use ImageCaptionLoader to load the image file
                    image_loader = ImageCaptionLoader(path_images=[file_path])

                    # Load image captions
                    image_documents = image_loader.load()

                    # Append the Langchain documents to the documents list
                    documents.extend(image_documents)
                    
                elif file_path.endswith((".pdf", ".docx", ".txt")):
                    # Use UnstructuredFileLoader to load the PDF/DOCX/TXT file
                    loader = UnstructuredFileLoader(file_path)
                    loaded_documents = loader.load()

                    # Extend the main documents list with the loaded documents
                    documents.extend(loaded_documents)

        # Load the website audio stream if URL is provided
        if website_url:
            loader = WebBaseLoader(website_url)
            web_data = loader.load()

            # Append the main documents list with the loaded documents
            documents.append(web_data[0])
            print(documents)

        # Chunk the data, create embeddings, and save in vectorstore
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
        document_chunks = text_splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_documents(document_chunks, embeddings)

        # Store the processed data in session state for reuse
        st.session_state.processed_data = {
            "document_chunks": document_chunks,
            "vectorstore": vectorstore,
        }

    else:
        # If the processed data is already available, retrieve it from session state
        document_chunks = st.session_state.processed_data["document_chunks"]
        vectorstore = st.session_state.processed_data["vectorstore"]

    # Initialize Langchain's QA Chain with the vectorstore
    qa = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever())

    # Initialize chat history if not already initialized
    st.session_state.messages = st.session_state.get("messages", [])

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask your questions?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Query the assistant using the latest chat history
        result = qa({"question": prompt, "chat_history": [(message["role"], message["content"]) for message in st.session_state.messages]})

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            full_response = result["answer"]
            message_placeholder.markdown(full_response + "|")
        message_placeholder.markdown(full_response)    
        print(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        
        # With a streamlit expander  
        with st.expander('Document Similarity Search'):
            # Find the relevant pages
            search = vectorstore.similarity_search_with_score(prompt) 
            # Write out the first 
            st.write(search[0][0].page_content) 
        message_placeholder.markdown(full_response) 

else:
    st.write("Please upload your files and provide a website URL for transcription.")