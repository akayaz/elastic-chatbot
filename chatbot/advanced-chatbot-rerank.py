import os
import getpass
import time
from typing import Dict

import elasticapm
import streamlit as st
from dotenv import load_dotenv
from langchain.globals import set_debug
from langchain_aws import ChatBedrock
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.chat_models import AzureChatOpenAI, ChatOpenAI, BedrockChat
from langchain_elasticsearch import ElasticsearchStore, ElasticsearchRetriever
from langchain.docstore.document import Document
from helper import get_conversational_rag_chain
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_openai import AzureChatOpenAI

set_debug(True)

load_dotenv()

ES_CID = os.getenv('CLOUD_ID')
ES_USER = os.getenv('CLOUD_USERNAME')
ES_PWD = os.getenv('CLOUD_PASSWORD')
ES_VECTOR_INDEX = os.getenv('LANGCHAIN_INDEX')
query_model_id = ".multilingual-e5-small_linux-x86_64"
OLLAMA_ENDPOINT = os.getenv('OLLAMA_ENDPOINT')

os.environ["COHERE_API_KEY"] = os.getenv('COHERE_API_KEY')

# load environment variables
load_dotenv()

# chat prompt
custom_template = """Given the following conversation and a follow-up message, \
    rephrase the follow-up message to a stand-alone question or instruction that \
    represents the user's intent, add all context needed if necessary to generate a complete and \
    unambiguous question or instruction, only based on the history, don't make up messages. \
    Maintain the same language as the original question.
    Use only the provided context to answer the question, if you don't know, simply answer that you don't know.

    Chat History:
    {chat_history}

    Follow Up Input: {question}
    Standalone question or instruction:"""

# streamlit UI Config
st.set_page_config(page_title="Elastic Chatbot", page_icon=":fr:", initial_sidebar_state="collapsed")
st.image(
    'https://images.contentstack.io/v3/assets/bltefdd0b53724fa2ce/blt601c406b0b5af740/620577381692951393fdf8d6'
    '/elastic-logo-cluster.svg',
    width=50)

st.header("Advanced Chatbot")
with st.sidebar:
    st.subheader('Choose LLM and parameters')
    st.write("Chatbot configuration")
    st.session_state.llm_model = st.sidebar.selectbox('Choose your LLM',
                                                      ['azure-openai', 'Ollama',
                                                       'bedrock'],
                                                      key='selected_model')
    if st.session_state.llm_model == 'Ollama':
        st.session_state.llm_base_url = st.sidebar.text_input('Ollama base url', OLLAMA_ENDPOINT)

    if st.session_state.llm_model == 'bedrock':
        model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
    st.session_state.llm_temperature = st.sidebar.slider('Temperature', min_value=0.0, max_value=1.0,
                                                         value=0.0,
                                                         step=0.1, key='llm_temp',
                                                         help='Control the creativity of the model')
    st.subheader('Configure Retrieval parameters')
    st.session_state.k = st.sidebar.slider('Number of documents to retrieve', min_value=1, max_value=20,
                                           value=10,
                                           step=1, key='k_results',
                                           help='Number of documents to retrieve')
    st.session_state.num_candidates = st.sidebar.slider('Number of candidates', min_value=20, max_value=200,
                                                        value=50,
                                                        step=1, key='num_of_candidates',
                                                        help='Number of candidates to use for vector search')
    st.session_state.rrf_window_size = st.sidebar.slider('RRF window size', min_value=50, max_value=200,
                                                         value=60,
                                                         step=10,
                                                         help='RRF window size')
    st.session_state.rrf_rank_constant = st.sidebar.slider('RRF rank constant', min_value=10, max_value=70,
                                                           value=20,
                                                           step=10,
                                                           help='RRF rank constant')
    st.session_state.display_sources = st.sidebar.checkbox('Display sources', value=True)
    st.session_state.rerank_results = st.sidebar.checkbox('Rerank results', value=False)

# test with streamlit context variables:
if 'rrf_window_size' not in st.session_state:
    st.session_state['rrf_window_size'] = 200
if 'rrf_rank_constant' not in st.session_state:
    st.session_state['rrf_rank_constant'] = 60
if 'k' not in st.session_state:
    st.session_state['k'] = 10
if 'num_candidates' not in st.session_state:
    st.session_state['num_candidates'] = 50

# global
rrf_window_size = st.session_state['rrf_window_size']
rrf_rank_constant = st.session_state['rrf_rank_constant']
k = st.session_state['k']
num_candidates = st.session_state['num_candidates']


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

vector_store = ElasticsearchStore(
    es_cloud_id=ES_CID,
    index_name=ES_VECTOR_INDEX,
    es_user=ES_USER,
    es_password=ES_PWD,
    query_field="content",
    vector_query_field="ml.inference.content.predicted_value",
    distance_strategy="DOT_PRODUCT",
    strategy=ElasticsearchStore.ApproxRetrievalStrategy(
        hybrid=True,
        query_model_id=query_model_id,
    )
)


def custom_document_builder(hit: Dict) -> Document:
    src = hit.get("_source", {})
    return Document(
        page_content=src.get("content", "Missing content!"),
        metadata={
            "title": src.get("title", "Missing title!"),
            "url": src.get("url", "Missing url!"),
            "id": src.get("id", "Missing id!"),
        },
    )


def custom_query_builder(query_body: dict, query: str):
    new_query_body: Dict = {
        "query": {
            "match": {
                "content": {
                    "query": query
                }
            }
        },
        "knn": {
            "field": "ml.inference.content.predicted_value",
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
                "window_size": rrf_window_size,
                "rank_constant": rrf_rank_constant
            }
        }
    }
    return new_query_body


def rrf_retriever(search_query: str) -> Dict:
    return {
        "retriever": {
            "rrf": {
                "retrievers": [
                    {
                        "knn": {
                            "field": "ml.inference.content.predicted_value",
                            "k": k,
                            "num_candidates": 50,
                            "query_vector_builder": {
                                "text_embedding": {
                                    "model_id": ".multilingual-e5-small_linux-x86_64",
                                    "model_text": search_query
                                }
                            }
                        }
                    },
                    {
                        "standard": {
                            "query": {
                                "match": {
                                    "content": {
                                        "query": search_query
                                    }
                                }
                            }
                        }
                    }
                ],
                "window_size": st.session_state.rrf_window_size,
                "rank_constant": st.session_state.rrf_rank_constant
            }
        },
        "size": k
    }


def hybrid_query(search_query: str) -> Dict:
    return {
        "query": {
            "match": {
                "content": {
                    "query": search_query
                }
            }
        },
        "knn": {
            "field": "ml.inference.content.predicted_value",
            "k": 5,
            "num_candidates": 50,
            "query_vector_builder": {
                "text_embedding": {
                    "model_id": ".multilingual-e5-small_linux-x86_64",
                    "model_text": search_query
                }
            }
        },
        "rank": {
            "rrf": {
                "window_size": 50,
                "rank_constant": 20
            }
        }

    }


def es_store_retriever(k, db, fetch_k):
    retriever = ElasticsearchRetriever.from_es_params(
        cloud_id=ES_CID,
        index_name=ES_VECTOR_INDEX,
        username=ES_USER,
        password=ES_PWD,
        body_func=rrf_retriever,
        document_mapper=custom_document_builder
    )
    return retriever


def basic_retriever(k, db, fetch_k):
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": k,
            "fetch_k": fetch_k,
            "custom_query": custom_query_builder,
            "doc_builder": custom_document_builder,
            "fields": ["title", "content", "url"],
        }
    )
    return retriever


def compressor_retriever(k, db, fetch_k):
    compressor = CohereRerank()
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=db.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": k,
                "fetch_k": fetch_k,
                "custom_query": custom_query_builder,
                "doc_builder": custom_document_builder,
            }
        )
    )
    return compression_retriever


def es_retriever_compressor_retriever(k, db, fetch_k):
    compressor = CohereRerank()
    compression_retriever: ContextualCompressionRetriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ElasticsearchRetriever.from_es_params(
            cloud_id=ES_CID,
            index_name=ES_VECTOR_INDEX,
            username=ES_USER,
            password=ES_PWD,
            body_func=rrf_retriever,
            document_mapper=custom_document_builder
        )
    )
    return compression_retriever


def main():
    # set up the retriever
    llm = None
    if st.session_state.llm_model == 'Ollama':
        llm = ChatOllama(base_url=st.session_state.llm_base_url,
                         model='mistral',
                         temperature=st.session_state.llm_temperature)

    if st.session_state.llm_model == 'bedrock':
        llm = ChatBedrock(model_id=model_id,
                          model_kwargs={"temperature": st.session_state.llm_temperature})
    if st.session_state.llm_model == 'azure-openai':
        os.environ["AZURE_OPENAI_API_KEY"] = os.getenv('AZURE_OPENAI_API_KEY')
        os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv('AZURE_OPENAI_ENDPOINT')
        llm = AzureChatOpenAI(
            temperature=st.session_state.llm_temperature,
            openai_api_version="2024-02-15-preview",
            azure_deployment="yazid-gpt4",
        )
    # init conversational rag chain
    if st.session_state.rerank_results:
        chain = get_conversational_rag_chain(llm=llm,
                                             retriever=es_retriever_compressor_retriever(st.session_state.k,
                                                                                         vector_store,
                                                                                         st.session_state.num_candidates),
                                             prompt_template=custom_template)
    else:
        chain = get_conversational_rag_chain(llm=llm,
                                             retriever=es_store_retriever(st.session_state.k,
                                                                          vector_store,
                                                                          st.session_state.num_candidates),
                                             prompt_template=custom_template)

    msgs = StreamlitChatMessageHistory()

    if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
        msgs.clear()
        msgs.add_ai_message("Bonjour! En quoi puis-je vous aider ?")

    avatars = {"human": "user", "ai": "assistant"}
    for msg in msgs.messages:
        st.chat_message(avatars[msg.type]).write(msg.content)

    if user_query := st.chat_input(placeholder="Dites moi ce que vous voulez savoir au sujet des actualités du "
                                               "service public..."):
        apm_client.begin_transaction("request")

        st.chat_message("user").write(user_query)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            try:
                elasticapm.label(query=user_query, llm=st.session_state.llm_model)
                response = chain({"question": user_query, "chat_history": msgs})
                for chunk in response['answer'].split():
                    full_response += chunk + " "
                    time.sleep(0.05)
                    message_placeholder.write(full_response)
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
                print(response['answer'])

                # print sources
                if st.session_state.display_sources and response['source_documents']:
                    print("---------------------------------------")
                    print(response['source_documents'])
                    print("---------------------------------------")
                    st.markdown(""" ##### Sources documents from the context: """)
                    for docs_source in response['source_documents']:
                        # st.write(docs_source.page_content)
                        link = f'<a href="{docs_source.metadata["url"]}" target="_blank">{docs_source.metadata["title"]}</a>'
                        st.markdown(link, unsafe_allow_html=True)

                elasticapm.set_transaction_outcome("success")
                apm_client.end_transaction("user-query-" + st.session_state.llm_model)

            except Exception as e:
                apm_client.capture_exception()
                elasticapm.set_transaction_outcome("failure")
                apm_client.end_transaction("user-query-" + st.session_state.llm_model)
                st.write("Oops! Error occurred while processing your request. Please try again later.")
                print(e)
                return


if __name__ == "__main__":
    main()
