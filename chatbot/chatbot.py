# import modules
from dotenv import load_dotenv
from langchain.vectorstores import ElasticsearchStore
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI, ChatVertexAI
import streamlit as st
import os
from langchain.prompts import PromptTemplate
from elasticsearch import Elasticsearch
import time
import elasticapm
import boto3
from langchain.llms import Bedrock
from langchain.llms import Ollama
from langchain.chat_models import AzureChatOpenAI
from typing import Dict
from langchain.globals import set_debug
from langchain.docstore.document import Document

set_debug(True)

# Load environment variables
load_dotenv()

st.set_page_config(
    page_title="Assistant",
    page_icon=":car:"
)
st.image(
    'https://images.contentstack.io/v3/assets/bltefdd0b53724fa2ce/blt601c406b0b5af740/620577381692951393fdf8d6/elastic-logo-cluster.svg',
    # "/Users/yakadiri/Downloads/stellantis-logo.png",
    width=100)
st.title("Assistant ESRE Demo")
st.sidebar.header('Configuration du Chatbot')
with st.sidebar:
    st.subheader('Choix du LLM et des paramètres')
    st.write("Paramètres du Chatbot")
    st.session_state.llm_model = st.sidebar.selectbox('Choisissez votre LLM',
                                                      ["azure-openai", 'Ollama', 'bedrock', ], key='selected_model',
                                                      help='Choisissez votre LLM')
    st.session_state.llm_temperature = st.sidebar.slider('Temperature', min_value=0.0, max_value=1.0, value=0.1,
                                                         step=0.1, key='llm_temp',
                                                         help='Controller la précision du modèle')
    st.subheader('Configure Retrieval parameters')
    st.session_state.k = st.sidebar.slider('Number of documents to retrieve', min_value=5, max_value=10,
                                           value=5,
                                           step=1, key='k_results',
                                           help='Number of documents to retrieve')
    st.session_state.num_candidates = st.sidebar.slider('Number of candidates', min_value=20, max_value=200,
                                                        value=50,
                                                        step=1, key='num_of_candidates',
                                                        help='Number of candidates to use for vector search')


@st.cache_resource
def initAPM():
    apm_client = elasticapm.Client(
        service_name=os.getenv('ELASTIC_APM_SERVICE_NAME'),
        server_url=os.getenv('ELASTIC_APM_SERVER_URL'),
        secret_token=os.getenv('ELASTIC_APM_SECRET_TOKEN'),
    )
    elasticapm.instrument()
    return apm_client


apm_client = initAPM()

# set elastic cloud id and password
CLOUD_ID = os.getenv('CLOUD_ID')
CLOUD_USERNAME = os.getenv('CLOUD_USERNAME')
CLOUD_PASSWORD = os.getenv('CLOUD_PASSWORD')
ES_VECTOR_INDEX = os.getenv('VECTOR_INDEX')
TRANSFORMER_MODEL_ID = os.getenv('EMBEDDING_MODEL_ID')

# Init bedrock client
session = boto3.Session(profile_name="default")
bedrock_client = session.client(service_name="bedrock-runtime")

# connect to es
es = Elasticsearch(
    cloud_id=CLOUD_ID,
    basic_auth=(CLOUD_USERNAME, CLOUD_PASSWORD)
)
es.info()

# set OpenAI API key
# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

vector_store = ElasticsearchStore(
    es_cloud_id=CLOUD_ID,
    es_user=CLOUD_USERNAME,
    es_password=CLOUD_PASSWORD,
    index_name=ES_VECTOR_INDEX,
    query_field='title',
    vector_query_field="ml.inference.body_content.predicted_value",
    distance_strategy="DOT_PRODUCT",
    strategy=ElasticsearchStore.ApproxRetrievalStrategy(query_model_id=TRANSFORMER_MODEL_ID,
                                                        hybrid=True)
)


def custom_query_builder(query_body: dict, query: str):
    new_query_body: Dict = {
        "query": {
            "match": {
                "title": {
                    "query": query
                }
            }
        },
        "knn": {
            "field": "ml.inference.body_content.predicted_value",
            "k": 5,
            "num_candidates": 50,
            "query_vector_builder": {
                "text_embedding": {
                    "model_id": ".multilingual-e5-small_linux-x86_64",
                    "model_text": query
                }
            }
        },
        "rank": {
            "rrf": {
                "window_size": 100,
                "rank_constant": 20
            }
        }
    }
    return new_query_body


def custom_document_builder(hit: Dict) -> Document:
    src = hit.get("_source", {})
    return Document(
        page_content=src.get("body_content", "Missing content!"),
        metadata={
            "title": src.get("title", "Missing title!"),
            # "director": src.get("director", "Missing director!"),
            # "year": src.get("release_year", "Missing year!"),
            "url": src.get("url", "Missing wiki page!"),
            # "release_year": src.get("release_year", "Missing release year!"),
        },
    )


# Init retriever
def init_retriever(k, db, fetch_k):
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": k,
            "fetch_k": fetch_k,
            "custom_query": custom_query_builder,
            "doc_builder": custom_document_builder,
            "fields": ["title", "body_content"],
        }

    )


# Build prompt for stuff chain
stuff_template = """
    Utilise uniquement les informations suivantes de contexte pour répondre à la question.  
    n'essaie pas d'inventer une réponse. Utilise au maximum 10 phrases. 
    Sois clair et détaillé dans tes réponses. 
    Termine tes réponses par "J'espere avoir répondu à votre question." sauf lorsque tu ne connais pas la réponse. 
{context}
Question: {question}
Helpful Answer:"""
stuff_prompt = PromptTemplate.from_template(stuff_template)

# Prompt for refine chain
template = """
    Utilise uniquement les informations suivantes de contexte pour répondre à la question. 
    n'essaie pas d'inventer une réponse. Utilise au maximum 10 phrases. 
    Sois clair et détaillé dans tes réponses. 
    Termine tes réponses par "J'espere avoir répondu à votre question." sauf lorsque tu ne connais pas la réponse. 
{context_str}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

refine_template = ("The original question is as follows: {question}\n"
                   "We have provided an existing answer: {existing_answer}\n"
                   "We have the opportunity to refine the existing answer(only if needed) with some more context below.\n"
                   "------------\n{context_str}\n------------\n"
                   "Given the new context, refine the original answer to better answer the question. "
                   "If the context isn't useful, always return the original answer."
                   "If you don't know the answer, say 'I don't know the answer to this question.'\n"
                   )

refine_prompt = PromptTemplate.from_template(refine_template)


def init_qa_chain(llm_model, llm_temperature, chain_type, retriever):
    llm = None
    if st.session_state.llm_model == 'gpt-3.5-turbo-16k':
        # set OpenAI API key
        llm = ChatOpenAI(
            model_name=st.session_state.llm_model,
            temperature=st.session_state.llm_temperature,
            model_kwargs={"engine": st.session_state.llm_model},
            openai_api_key=os.getenv('OPENAI_API_KEY'))

    if st.session_state.llm_model == 'azure-openai':
        os.environ["AZURE_OPENAI_API_KEY"] = os.getenv('AZURE_OPENAI_API_KEY')
        os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv('AZURE_OPENAI_ENDPOINT')
        llm = AzureChatOpenAI(
            temperature=st.session_state.llm_temperature,
            openai_api_version="2023-07-01-preview",
            azure_deployment="yazid-gpt4",
            # azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
        )

    # aws bedrock
    if st.session_state.llm_model == 'bedrock':
        default_model_id = "anthropic.claude-v2"
        AWS_MODEL_ID = default_model_id
        llm = Bedrock(
            client=bedrock_client,
            model_id=AWS_MODEL_ID
        )

    if st.session_state.llm_model == 'Ollama':
        llm = Ollama(
            model="mistral"
        )

    if chain_type == "stuff":
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": stuff_prompt},
            verbose=True
        )
    if chain_type == "refine":
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="refine",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "question_prompt": QA_CHAIN_PROMPT,
                "refine_prompt": refine_prompt},
            verbose=True
        )


def main():
    retriever = init_retriever(st.session_state.k, vector_store, st.session_state.num_candidates)

    default_query: str = "Quelle est la politique de qualité de l'AIFE?"
    if question := st.text_input("Posez votre question", value=default_query):
        # start APM Transaction
        apm_client.begin_transaction("request")
        try:
            elasticapm.label(query=question, llm=st.session_state.llm_model)
            qa = init_qa_chain(llm_model=st.session_state.llm_model, llm_temperature=st.session_state.llm_temperature,
                               chain_type="stuff", retriever=retriever)

            with st.chat_message("assistant"):
                st.spinner("Recherche de la réponse...")
                message_placeholder = st.empty()
                full_response = ""
                # stream_handler = StreamHandler(st.empty())
                with elasticapm.capture_span("qa-" + st.session_state.llm_model, "qa"):
                    response = qa({"query": question})

                    for chunk in response['result'].split():
                        full_response += chunk + " "
                        time.sleep(0.05)
                        message_placeholder.write(full_response)
                        # Add a blinking cursor to simulate typing
                        message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
                st.markdown(""" ##### Sources: """)

                for docs_source in response['source_documents']:
                    # st.markdown(" %s " %docs_source.metadata['url'])
                    link = f'<a href="{docs_source.metadata["url"]}" target="_blank">{docs_source.metadata["title"]}</a>'
                    st.markdown(link, unsafe_allow_html=True)
                # st.markdown(""" ###### LLM: """ + st.session_state.llm_model)
            elasticapm.set_transaction_outcome("success")
            apm_client.end_transaction("user-query-" + st.session_state.llm_model)
        except Exception as e:
            apm_client.capture_exception()
            elasticapm.set_transaction_outcome("failure")
            apm_client.end_transaction("user-query-" + st.session_state.llm_model)
            print(e)
            st.error(e)


if __name__ == '__main__':
    main()
