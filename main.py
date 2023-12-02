import os
import streamlit as st
import pickle
import time
import langchain
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import SeleniumURLLoader
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from PIL import Image

from dotenv import load_dotenv

load_dotenv('.env')

os.environ['OPENAI_API_KEY'] = "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"

def set_background_image(image_path):
    # Set the background image using custom CSS
    st.markdown(
        f"""
        <style>
            body {{
                background-image: url('{image_path}');
                background-size: cover;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

image = Image.open("e:\Chatbot_Langchain\download.png")
st.image(image, caption="Your favorite freind to find insights from long articles", use_column_width=True)
set_background_image(image)


st.title("InfoBOT : News research Tool")

st.sidebar.title("News article URL's")

urls = []

for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_urls_clicked = st.sidebar.button("Process URLs")

main_placeholder = st.empty()

llm = OpenAI(temperature=0.9, max_tokens=500) 

if process_urls_clicked:
    st.write("Processing the URLs")
    #loading data 
    loader = SeleniumURLLoader(urls = urls)
    main_placeholder.text("Data Loading has started")
    data = loader.load()


    #splitting data
    text_splitter = RecursiveCharacterTextSplitter(
    separators = ['\n\n', '\n',',','.'],
    chunk_size=1000
    )
    main_placeholder.text("Text splitting has started")
    docs = text_splitter.split_documents(data)


    #embeddings and saving data to FAISS index
    embedding = OpenAIEmbeddings()
    st.write(type(embedding))
    vectorstore_openai = FAISS.from_documents(docs,embedding)
    main_placeholder.text("Embedding Vector started building")
    time.sleep(2)
    vectorstore_openai.save_local("faiss_indexes")



query = main_placeholder.text_input("Question:  ")

if query:
    vectorIndex = FAISS.load_local("faiss_indexes", embedding)
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorIndex.as_retriever())
    st.write("Calling LLM")
    result = chain({"question": query}, return_only_outputs=True)
    
    
    #Display the answer
    st.subheader("Answer: ")
    st.write(result["answer"])

    #Display sources if available
    sources = result.get("sources", "")
    if sources:
        st.subheader("Sources: ")
        sources_list = sources.split("\n")
        for source in sources_list:
            st.write(source)

else:
    st.error("No existing knowledge found")