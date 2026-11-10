import streamlit as st
import json
import tempfile
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Define data models
class Product(BaseModel):
    name: str | None
    description: str | None
    quantity: int
    date: str
    total: float

class PoResponse(BaseModel):
    po_number: str
    items_order: list[Product]
    subtotal: float
    total: float

# Setup page configuration
st.set_page_config(
    page_title="PO Analyzer",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .upload-section {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
    }
    .results-section {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .header {
        color: #1E3A8A;
        margin-bottom: 1.5rem;
    }
    .subheader {
        color: #4B5563;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.markdown("<h1 class='header'>📋 Kaeso PO Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>Upload and analyze purchase orders with AI-powered extraction</p>", unsafe_allow_html=True)

# Main function to process PO
def process_po(file_path):
    with st.spinner("Processing your PO... This may take a moment."):
        # Initialize LLM
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        # Load and process the document
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        
        # Create embeddings and vector store
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        text_splitter = RecursiveCharacterTextSplitter()
        documents = text_splitter.split_documents(docs)
        vector = FAISS.from_documents(documents, embeddings)
        
        # Set up retriever
        retriever = vector.as_retriever(
            search_type='similarity',
            search_kwargs={"k": 10}
        )
        
        # Create prompt templates and chains
        history_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            (
                "user",
                "Given the above conversation, generate a search query to look up to get information relevant to the conversation",
            ),
        ])
        
        retriever_chain = create_history_aware_retriever(llm, retriever, history_prompt)
        
        extraction_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
                You are Kaesia, the best PO analyzer in the world. Your role is to analyze and extract PO information.
                Extract all information in the specified structured format with client_name, po_number, items_order (list of products),date,subtotal, and total.
                
                Return only JSON without any markdown formatting or code blocks.
                
                Context:\n\n{context}
                """,
            ),
            ("user", "{input}"),
        ])
        
        document_chain = create_stuff_documents_chain(llm, extraction_prompt)
        retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)
        
        response = retrieval_chain.invoke({
            "input": "Extract all details from this PO in structured format. Return only valid JSON without markdown.",
        })
        
        try:
            response_text = response["answer"]
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
                
            json_data = json.loads(response_text)
            return json_data
        except json.JSONDecodeError as e:
            st.error(f"Error parsing JSON response: {e}")
            st.text(response["answer"])
            return None

# Main app layout
with st.container():
    
    uploaded_file = st.file_uploader("Upload your PO (PDF format)", type="pdf")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Create a temporary file to save the uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            # Process the uploaded PO
            po_data = process_po(tmp_path)
            
            if po_data:
                st.markdown("<div class='results-section'>", unsafe_allow_html=True)
                
                # Display PO information
                col1, col2 = st.columns(2)
                
                with col1:
                    if "client_name" in po_data:
                        st.markdown(f"**Client:** {po_data['client_name']}")
                    st.markdown(f"**PO Number:** {po_data.get('po_number', 'N/A')}")
                    st.markdown(f"**Date:** {po_data.get('date', 'N/A')}")
                
                with col2:
                    st.markdown(f"**Subtotal:** ${po_data.get('subtotal', 0.0):.2f}")
                    st.markdown(f"**Total:** ${po_data.get('total', 0.0):.2f}")
                
                # Display items in the order
                st.markdown("### Items in Order")
                
                items = po_data.get("items_order", [])
                if items:
                    # Create a table for items
                    items_data = []
                    for item in items:
                        item_dict = {
                            "Product": item.get("name", "N/A"),
                            "Description": item.get("description", "N/A"),
                            "Quantity": item.get("quantity", 0),
                            "Price": f"${item.get('price', 0.0):.2f}"
                        }
                        items_data.append(item_dict)
                    
                    st.table(items_data)
                else:
                    st.info("No items found in the PO.")
                
                # Show raw JSON data in expandable section
                with st.expander("View Raw JSON Data"):
                    st.json(po_data)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Download option for extracted data
                json_str = json.dumps(po_data, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name=f"po_{po_data.get('po_number', 'extract')}.json",
                    mime="application/json"
                )
            
        finally:
            # Clean up the temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    else:
        st.info("Upload a PO file to begin analysis")
        st.markdown("</div>", unsafe_allow_html=True)

# Add footer
st.markdown("""
<div style="text-align: center; margin-top: 3rem; color: #6B7280; font-size: 0.8rem;">
    Powered by KAESO PO Analyzer | © 2025
</div>
""", unsafe_allow_html=True)