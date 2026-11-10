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
def analise(doc,role):
    llm=ChatOpenAI(model="gpt-4o-mini",temperature=0)
    llm_with_structured_output=llm.with_structured_output(PoResponse)
    loader = PyPDFLoader(doc)
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
            You are Kaesia, the best CV analyzer in the world. Your task is to analyze the resume provided in the context
            and extract relevant information based on the specified role.

            Role to analyze: {role}
            Guidelines:
            - Be strict in your evaluation. Do not assume or infer information that is not explicitly stated.
            - Give higher weight to work experience that is clearly related to the role, compared to general skills or unrelated experience.
            - Always balance the candidate’s technical skills with their hands-on experience.
            - Evaluate whether the candidate has directly worked in or has practical experience with responsibilities aligned to the role.
            - Include the companies the candidate has worked for and calculate the total years of relevant experience based on the job history.
            - Highlight both strengths and areas for improvement with respect to the role requirements.


            Carefully read the resume below and extract the information in the following structured format:

            {{
                "name": "",
                "email": "",
                "phone": "",
                "education": [],
                "work_experience": [
                    {{
                        "company": "",
                        "position": "",
                        "start_date": "",
                        "end_date": "",
                        "responsibilities": "",
                        "related_to_role": true | false
                    }}
                ],
                "projects": [
                    {{
                        "name": "",
                        "description": "",
                        "technologies_used": "",
                        "related_to_role": true | false
                    }}
                ],
                "certifications": [
                    {{
                        "name": "",
                        "issuing_organization": "",
                        "date_issued": "",
                        "related_to_role": true | false
                    }}
                ],
                "technical_skills": [
                    {{
                        "skill": "",
                        "proficiency": "",
                        "related_to_role": true | false
                    }}
                ],
                "languages": [
                    {{
                        "language": "",
                        "proficiency": ""
                    }}
                ],
                "role_fit_analysis": {{
                    "strengths": [
                        {{
                            "description": "",
                            "related_to_role": true | false
                        }}
                    ],
                    "areas_for_improvement": [
                        {{
                            "description": "",
                            "related_to_role": true | false
                        }}
                    ],
                    "score": "from 0 to 100"
                }}
               
            }}

            Resume context:
            {context}
            """
        ),
        ("user", "{input}"),
    ]
)
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

    response = retrieval_chain.invoke(
        {"input": "Extract all details from this PO in structured format","role": role},
    )
    print("Processando, aguarde um pouco....")
    json_data = json.loads(response["answer"])
    json_response = json.dumps(json_data, indent=2)
    return json_response