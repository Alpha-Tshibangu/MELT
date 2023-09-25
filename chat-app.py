import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
import os

# Chat UI title
st.header("Machine Enabled Legislation Transposition")

# File uploader in the sidebar on the left
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.")
    st.stop()

# Set OPENAI_API_KEY as an environment variable
os.environ["OPENAI_API_KEY"] = openai_api_key

llm = ChatOpenAI(temperature=0.1, model_name="gpt-4",streaming=True)

# Load version history from the text file
def load_version_history():
    with open("version_history.txt", "r") as file:
        return file.read()

# Sidebar section for uploading files and providing a YouTube URL
with st.sidebar:
    uploaded_files = st.file_uploader("Please upload your files", accept_multiple_files=True, type=None)

    # Create an expander for the version history in the sidebar
    with st.sidebar.expander("**Version History**", expanded=False):
        st.write(load_version_history())

    st.info("Please refresh the browser if you decide to upload more files to reset the session", icon="🚨")

with st.spinner('Wait for it...'):
    # Check if files are uploaded
    if uploaded_files:
        # Print the number of files to console
        st.write(f"Number of files uploaded: {len(uploaded_files)}")
        print(f"Number of files uploaded: {len(uploaded_files)}")

        # Load the data and perform preprocessing only if it hasn't been loaded before
        if "processed_data" not in st.session_state:
            # Load the data from uploaded PDF files
            documents = []
            for uploaded_file in uploaded_files:
                # Get the full file path of the uploaded file
                file_path = os.path.join(os.getcwd(), uploaded_file.name)

                # Save the uploaded file to disk
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                # Use UnstructuredFileLoader to load the PDF file
                loader = PyPDFLoader(file_path)
                loaded_documents = loader.load_and_split()
                print(f"Number of files loaded: {len(loaded_documents)}")

                # Extend the main documents list with the loaded documents
                documents.extend(loaded_documents)

            embeddings = OpenAIEmbeddings()
            vectorstore = Chroma.from_documents(loaded_documents, embeddings)

            # Store the processed data in session state for reuse
            st.session_state.processed_data = {
                "document_chunks": loaded_documents,
                "vectorstore": vectorstore,
            }

            # Print the number of total chunks to console
            print(f"Number of total chunks: {len(loaded_documents)}")

        else:
            # If the processed data is already available, retrieve it from session state
            document_chunks = st.session_state.processed_data["document_chunks"]
            vectorstore = st.session_state.processed_data["vectorstore"]

        # Initialize Langchain's QA Chain with the vectorstore
        qa = ConversationalRetrievalChain.from_llm(llm,vectorstore.as_retriever())

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

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
                
            # With a streamlit expander  
            with st.expander('Document Similarity Search'):
                # Find the relevant pages
                search = vectorstore.similarity_search_with_score(prompt) 
                # Write out the first 
                st.write(search[0][0].page_content) 
            message_placeholder.markdown(full_response) 
            
            print(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

    else:
        st.write("Please upload your files.")
