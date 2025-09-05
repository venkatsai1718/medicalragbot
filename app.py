import streamlit as st
import numpy as np
from pypdf import PdfReader
import tiktoken
import faiss
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


API_KEY = st.secrets['google_api_key']


chat_model = ChatGoogleGenerativeAI(api_key=API_KEY, model='gemini-1.5-flash', temparature=0.2)

embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=API_KEY, model="models/embedding-001")

def read_pdf(file):
    try:
        reader = PdfReader(file)
        return "\n".join([page.extract_text() or "" for page in reader.pages])
    except Exception as e:
        print(f"[ERROR] while reading PDF: {str(e)}")

def chunk_text(text):
    max_tokens = 30
    tokenizer = tiktoken.get_encoding("cl100k_base")
    words = text.split()
    chunks, chunk, tokens = [], [], 0
    for word in words:
        token_count = len(tokenizer.encode(word))
        if tokens + token_count > max_tokens:
            chunks.append(" ".join(chunk))
            chunk, tokens = [word], token_count
        else:
            chunk.append(word)
            tokens += token_count
    if chunk:
        chunks.append(" ".join(chunk))
    return chunks

def get_embedding(text):
    vector = embedding_model.embed_query(text)
    return vector

def build_index(chunks):
    embeddings = [get_embedding(chunk) for chunk in chunks]
    dims = len(embeddings[0])
    index = faiss.IndexFlatL2(dims)
    index.add(np.array(embeddings))
    return index


def handle_query(query, index, chunks):
    query_emb = np.array(get_embedding(query)).reshape(1, -1)
    distances, indices = index.search(query_emb, k=4)
    relavent_chunks = [chunks[i] for i in indices[0]]
    context = "\n\n".join(relavent_chunks)
    
    prompt = PromptTemplate(
        template="""
        You are an helpful assistant to assist healthcare professionals.
        
        Context:
        {context}
        
        Question:
        {query}
        
        Answer:""",
        input_variables=['context', 'query']
    )
    
    parser = StrOutputParser()
    chain = prompt | chat_model | parser
    response = chain.invoke({"query": query, 'context': context})
    return response

def Ui():
    st.markdown("<h3 style='text-align:center;color: orangered;'>📄 Medical Assistant</h3>", unsafe_allow_html=True)
    
    pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    
    chunks = []
    faiss_index = None
    
    if pdf_file:
        text = read_pdf(pdf_file)
        chunks = chunk_text(text)
        faiss_index = build_index(chunks)

        st.success(f"Document loaded and processed successfully !!")
    
    if faiss_index:
        question = st.text_input("Ask a question about the document:")
        if st.button("Send") and question:
            answer = handle_query(question, faiss_index, chunks)
            
            st.markdown(f"**Answer:\n **{answer}")

Ui()