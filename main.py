#llm = ChatGroq(temperature=0, model_name="gemma2-9b-it")
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pydantic import BaseModel
import json
load_dotenv()

class Product(BaseModel):
    name:str|None
    description:str|None
    quantity:int
    date:str
    total:float

class PoResponse(BaseModel):
    po_number:str
    items_order:list[Product]
    subtotal:float
    total:float

llm=ChatOpenAI(model="gpt-4o-mini",temperature=0)
llm_with_structured_output=llm.with_structured_output(PoResponse)
loader = PyPDFLoader('./docs/po2.pdf')
docs = loader.load()
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)
retriever = vector.as_retriever(
    search_type='similarity',
    search_kwargs={"k":10}
)

prompt = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        (
            "user",
            "Given the above conversation, generate a search query to look up to get information relevant to the conversation",
        ),
    ]
)
retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are Kaesia, the best PO analyzer in the world. Your role is to analyze and extract PO information.
            Extract all information in the specified structured format with po_number, items_order (list of products), subtotal, and total.
            

            :\n\n{context}""",
                    ),
                    ("user", "{input}"),
                ]
)
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

response = retrieval_chain.invoke(
    {"input": "Extract all details from this PO in structured format"}
)
print("Processando, aguarde um pouco....")
json_data = json.loads(response["answer"])
json_response = json.dumps(json_data, indent=2)
print(json_response)