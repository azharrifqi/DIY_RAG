from langchain.chains import RetrievalQA
import vertexai
from langchain_google_vertexai import VertexAI
from vertexai.generative_models import GenerativeModel, Part, Content, FinishReason, SafetySetting
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain.chains import create_retrieval_chain
from langchain_core.messages import HumanMessage
from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.embeddings import OpenAIEmbeddings
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores.utils import filter_complex_metadata
import streamlit as st
import os
from langchain.prompts import ChatPromptTemplate
from google.cloud import bigquery
from streamlit.runtime.scriptrunner import get_script_run_ctx
from google.cloud.sql.connector import Connector
import vertexai.preview.generative_models as generative_models
import sqlalchemy
import google.cloud.logging
from google.cloud import secretmanager
import json
from sqlalchemy.dialects.postgresql import UUID
import uuid

#Check Connection
log_client = google.cloud.logging.Client()
log_client.setup_logging()

PROJECT_ID = "844021890758"
COLLECTION_NAME = "default_collection"
SECRET_ID_DB = "db-secret"

ctx = get_script_run_ctx()
session_global = ctx.session_id


client = bigquery.Client()

rag_chain = None
# Define the Chroma directory
CHROMA_DB_DIR = "chroma_db_streamlit"


def load_data_and_create_vector_db():
    query = """
        SELECT product_name, description, retail_price, product_rating, overall_rating, string(DATE(crawl_timestamp)) as crawl_timestamp
        FROM `dla-gen-ai-specialization.demo_dataset_1.e_commerce`
        LIMIT 5000
    """
    query_job = client.query(query)
    raw_results = query_job.result()

    # get as document
    documents = []

    for row in raw_results:
        # Extract `product_name` as the page content
        page_content = row["product_name"]

        # Prepare metadata with all other columns
        metadata = {
            key: value
            for key, value in dict(row).items()
            if key != "product_name"  # Exclude the `product_name` column
        }
        
        documents.append(Document(page_content=page_content, metadata=metadata))
    return documents

# Function to set up your RAG system
@st.cache_resource
def gemini_main():
    global rag_chain
    if os.path.exists(CHROMA_DB_DIR):
        
        print("Loading existing Chroma DB...")
        # Load the existing database
        embeddings = VertexAIEmbeddings(model_name="text-embedding-005")
        vector_db = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
        
        retriever = vector_db.as_retriever()
        
        project_id = "dla-gen-ai-specialization"

        llm = VertexAI(
            temperature=0.5,
            model="gemini-1.5-pro-002" 
        )
        vertexai.init(project=project_id, location='asia-southeast2')
        
        system_prompt = (""" You are an AI agent designed to assist users with product inquiries. Follow these steps to answer {context} meticulously to provide accurate and helpful information:

        1. **Description Analysis**: Extract detailed information from the product description. Focus on elements such as price, color, usage, specifications, and features.

        2. **Price Extraction**: Identify and store the retail price of the product in a variable called `retail_price`.

        3. **Rating Inquiry**:
        - If the user inquires about ratings, extract `product_rating` and `overall_rating`.
        - Classify the `product_rating` as follows:
        - If `product_rating` is one of '4.4', '4.1', '4.6', or '4.9', respond that the product is rated above '4'.
        - If `product_rating` is one of '3.2', '3.5', or '3.7', respond that the product is rated above '3'.
        - Ensure to provide a product that meets the rating criteria when asked (e.g., "What product has a rating above 4?").

        4. **Product Recommendation**:
        - When the user requests recommendations for good products, filter through the available products.
        - Sort the filtered products first by `retail_price` and then by `product_rating`.

        5. **Recommendation Summary**: When recommending a product, provide a clear summary detailing why this product is recommended. Utilize details from the description, ratings, 
        or pricing to justify the recommendation. make it with 1 sentence for every recommended product

        Utilize this structured approach to ensure clarity, relevance, and precision in your responses.
                        
        Do not asked more information, only answer with Recommendation Summary
                        
        for example
        question : "What recommend shoes for Kids?"
        your answer :
            - Ajanta SkolAr Casuals: These white casual shoes are designed for boys, featuring a bacteria-free cotton lining, wide toe, and impact protection toe, making them suitable for school and casual wear. Rating:
            - Zebra Outdoors: These golden sports shoes are designed for girls, featuring a rubber sole, velcro closure, and mesh inner material, making them suitable for sports activities. Rating
        
        Give additional answer with rating or retail_price
        """)

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        question = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever,question)
        
        return rag_chain
    else:
        print("Creating new Chroma DB...")
        # Setup LLM
        project_id = "dla-gen-ai-specialization"

        llm = VertexAI(
                temperature=0.5,
                model="gemini-1.5-pro-002"        
            )
        vertexai.init(project=project_id, location='asia-southeast2')

        # Chunking
        documents = load_data_and_create_vector_db()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        split_docs = text_splitter.split_documents(documents)
        len(split_docs)

        docs = filter_complex_metadata(split_docs)

        # Setup Embedding
        embeddings = VertexAIEmbeddings(model_name="text-embedding-005")
    
        vector_db = Chroma.from_documents(documents=docs, embedding=embeddings ,persist_directory='chroma_db_streamlit')

        retriever = vector_db.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        
        system_prompt = ("""
            You are an AI agent designed to assist users with product inquiries. Follow these steps to answer {context} meticulously to provide accurate and helpful information:

        1. **Description Analysis**: Extract detailed information from the product description. Focus on elements such as price, color, usage, specifications, and features.

        2. **Price Extraction**: Identify and store the retail price of the product in a variable called `retail_price`.

        3. **Rating Inquiry**:
        - If the user inquires about ratings, extract `product_rating` and `overall_rating`.
        - Classify the `product_rating` as follows:
        - If `product_rating` is one of '4.4', '4.1', '4.6', or '4.9', respond that the product is rated above '4'.
        - If `product_rating` is one of '3.2', '3.5', or '3.7', respond that the product is rated above '3'.
        - Ensure to provide a product that meets the rating criteria when asked (e.g., "What product has a rating above 4?").

        4. **Product Recommendation**:
        - When the user requests recommendations for good products, filter through the available products.
        - Sort the filtered products first by `retail_price` and then by `product_rating`.

        5. **Recommendation Summary**: When recommending a product, provide a clear summary detailing why this product is recommended. Utilize details from the description, ratings, 
        or pricing to justify the recommendation. make it with 1 sentence for every recommended product

        Utilize this structured approach to ensure clarity, relevance, and precision in your responses.
                        
        Do not asked more information, only answer with Recommendation Summary
                        
        for example
        question : "What recommend shoes for Kids?"
        your answer :
            - Ajanta SkolAr Casuals: These white casual shoes are designed for boys, featuring a bacteria-free cotton lining, wide toe, and impact protection toe, making them suitable for school and casual wear. Rating:
            - Zebra Outdoors: These golden sports shoes are designed for girls, featuring a rubber sole, velcro closure, and mesh inner material, making them suitable for sports activities. Rating
        
        Give additional answer with rating or retail_price
        """)

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )
            
        question = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever,question)
        
        return rag_chain


@st.cache_resource
def gemini_paraphrase():
    vertexai.init(project=PROJECT_ID, location="asia-southeast1")
    model = GenerativeModel(
        "gemini-1.5-pro-002",
        generation_config=generation_config
    )
    return model

# Main function to run the RAG system
def get_answer(retriever, question):
    return retriever.invoke({"input": question})


st.title("Demo 1 Gen AI Specialization")
generation_config = {
    "max_output_tokens": 8192,
    "temperature": 1,
    "top_p": 0.95,
}

safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_NONE,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_NONE,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
}

@st.cache_resource
def access_secret():
    # secret manager
    client = secretmanager.SecretManagerServiceClient()
    # Build the resource name of the secret version.
    name = f"projects/{PROJECT_ID}/secrets/{SECRET_ID_DB}/versions/latest"

    # Access the secret version.
    response = client.access_secret_version(request={"name": name})
    payload = response.payload.data.decode("UTF-8")
    db_secret = json.loads(payload)
    return db_secret


def store_feedback(response, feedback_type, suggestion=None):
    feedback_entry = (
        # f"Timestamp: {datetime.now().isoformat()}\n"
        f"Response: {response}\n"
        f"Feedback Type: {feedback_type}\n"
    )
    if suggestion:
        feedback_entry += f"User Suggestion: {suggestion}\n"
    feedback_entry += f"{'-'*40}\n"
    
    # Save feedback to a .txt file
    with open("feedback_log.txt", "a") as file:
        file.write(feedback_entry)

# ai_response = None

# global session for retrieving memory from db
if not "initialized" in st.session_state:
    st.session_state['session'] = session_global

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize feedback
if "get_feedback" not in st.session_state:
    st.session_state.get_feedback = False

# if "feedback_value" not in st.session_state:
    # st.session_state.feedback_value = None

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        # db connection
        connector = Connector()
        db_secret = access_secret()
        def getconn():
            conn = connector.connect(
                db_secret['INSTANCE_CONNECTION_NAME'],
                "pg8000",
                user=db_secret['DB_USER'],
                password=db_secret['DB_PASS'],
                db=db_secret['DB_NAME']
            )
            return conn

        pool = sqlalchemy.create_engine(
        "postgresql+pg8000://",
        creator=getconn,
        )
        gemini_main_model = gemini_main()
        gemini_paraphrase_model = gemini_paraphrase()

        # retrieve chat history
        history = []
        with pool.connect() as db_conn:
            # returning vacancy information for prompting
            query_history_data = f"""SELECT user_chat, ai_chat FROM demo_1_chat where session_id = '{session_global}' ORDER BY inserttimestamp DESC LIMIT 3"""
            query_history_result = db_conn.execute(sqlalchemy.text(query_history_data)).fetchall()
            for row in query_history_result:
                rows = row._mapping
                row_as_dict = dict(rows)
                history_temp = {
                    "user": row_as_dict['user_chat'],
                    "ai": row_as_dict['ai_chat']
                }
                history.append(history_temp)
        connector.close()
        if history == []:
            search_paraphrase = prompt
        else:
            paraphrase_prompt = f""""
            Adjust the following question based on the historical chat. If the question doesn't relate to the historical chat, no adjustment is needed.
            
            here is the question '{prompt}'
            
            here is the historical chat
            {history}"""
            responses = gemini_paraphrase_model.generate_content(
                [paraphrase_prompt],
                safety_settings=safety_settings,
                stream=False
            )
            search_paraphrase = responses.text

        responses = get_answer(gemini_main_model, prompt)
        hasil = []
        def streaming():
        #     for response in responses:
        #         # hasil.append(response.text)
        #         hasil.append(response)
        #         yield response
            response_text = responses.get("answer")
            hasil.append(response_text)
            yield response_text
        st.write_stream(streaming)
        hasil = ''.join(hasil)

        # ai_response = hasil

        # update data on db
        insert_data = sqlalchemy.text("""INSERT INTO demo_1_chat (session_id, user_chat, ai_chat) values (:session_id, :user_chat, :ai)""")
        with pool.connect() as db_conn:
            # update job information
            bind_params = [
                sqlalchemy.sql.bindparam(key="session_id", value=uuid.UUID(session_global), type_=UUID(as_uuid=True)),
                sqlalchemy.sql.bindparam(key="user_chat", value=prompt),
                sqlalchemy.sql.bindparam(key="ai", value=hasil),
            ]
            db_conn.execute(insert_data.bindparams(*bind_params))
            db_conn.commit()
        connector.close()
    # hasil_store = hasil
    st.session_state.messages.append({"role": "assistant", "content": hasil})
    st.session_state.get_feedback = True

    ai_response = hasil
    st.session_state["ai_response"] = hasil

def save_feedback(feedback_result):
    
    if "ai_response" in st.session_state:
        ai_response = st.session_state["ai_response"]

    if feedback_result == 1:
        store_feedback(ai_response, "thumb_up")
        st.success("Thank you for your positive feedback! It has been recorded.")

    elif feedback_result == 0:
        # feedback_text = st.chat_input("Write your feedback here which consist of correct answer", "Feedback")
        st.write("### Please suggest how the AI should respond:")
        user_suggestion = st.text_input("Your suggestion:")
        
        if st.button("Submit Feedback"):
            if user_suggestion.strip():
                store_feedback(ai_response, "thumb_down", user_suggestion)
                st.success("Thank you for your feedback! It has been recorded.")
            else:
                st.warning("Please provide a suggestion before submitting.")

        st.session_state.get_feedback = False

# if st.session_state.get_feedback:
feedback_result = st.feedback("thumbs")

if feedback_result is not None:  # Ensure feedback is provided
    save_feedback(feedback_result)