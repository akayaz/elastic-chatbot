from langchain.vectorstores.elasticsearch import ElasticsearchStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import NLTKTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
from elasticsearch import Elasticsearch
import json
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import JSONLoader

load_dotenv()

CLOUD_ID = os.getenv('CLOUD_ID')
CLOUD_USERNAME = os.getenv('CLOUD_USERNAME')
CLOUD_PASSWORD = os.getenv('CLOUD_PASSWORD')
INDEX_NAME = os.getenv('ES_VECTOR_INDEX')
EMBEDDING_MODEL_ID = os.getenv('EMBEDDING_MODEL_ID')
DATASET = os.getenv('DATASET_PATH')

query_model_id = ".multilingual-e5-small_linux-x86_64"
INGEST_PIPELINE_ID = "stellantis-rag-langchain"

# ingest pipeline processors
ingest_pipeline_processors = {
    "processors": [
        {
            "inference": {
                "model_id": query_model_id,
                "field_map": {
                    "query_field": "text_field"
                },
                "target_field": "vector_query_field",
                "on_failure": [
                    {
                        "append": {
                            "field": "_source._ingest.inference_errors",
                            "value": [
                                {
                                    "message": "Processor 'inference' in pipeline 'movies-rag-langchain-pipeline' "
                                               "failed with message '{{"
                                               "_ingest.on_failure_message }}'",
                                    "pipeline": INGEST_PIPELINE_ID,
                                    "timestamp": "{{{ _ingest.timestamp }}}"
                                }
                            ]
                        }
                    }
                ]
            }
        }
    ]
}


def create_index(client, index_name):
    """Creates an index in Elasticsearch if one isn't already there."""
    if not client.indices.exists(index=INDEX_NAME):
        pass
    else:
        print("Deleting existing movies index...")
        client.options(ignore_status=[404, 400]).indices.delete(index=INDEX_NAME)
    # index mapping
    mappings = {
        "properties": {
            "title": {
                "type": "text"
            },
            "body_content": {
                "type": "text"
            },
            "text_field": {
                "type": "text"
            },
            "url": {
                "type": "keyword"
            },
            "vector_query_field": {
                "properties": {
                    "is_truncated": {
                        "type": "boolean"
                    },
                    "predicted_value": {
                        "type": "dense_vector",
                        "dims": 384,
                        "index": True,
                        "similarity": "dot_product"
                    }
                }
            }
        }
    }
    client.indices.create(index=index_name,
                          body={
                              "mappings": mappings,
                              "settings": {
                                  "index": {
                                      "default_pipeline": INGEST_PIPELINE_ID
                                  }
                              }
                          })

    # index setting
    settings = {
        "analysis": {
            "analyzer": {
                "my_english_analyzer": {
                    "type": "english",
                    "stopwords": ["_english_", "where", "which", "how", "when", "wherever"]
                }
            }
        }, "default_pipeline": INGEST_PIPELINE_ID
    }
    client.options(ignore_status=[400, 404]).indices.create(
        index=index_name,
        settings=settings,
        mappings=mappings
    )


def create_ingest_pipeline(client, pipeline_id):
    # Check if the pipeline exists
    if client.options(ignore_status=[404, 400]).ingest.get_pipeline(id=pipeline_id):
        client.options(ignore_status=[404, 400]).ingest.delete_pipeline(id=pipeline_id)
    # Create the pipeline
    print("Creating ingest pipeline...")
    client.options(ignore_status=[400]).ingest.put_pipeline(
        id=pipeline_id,
        description="Ingest pipeline to generate plot_vector",
        processors=ingest_pipeline_processors['processors'])
    print(f"Pipeline {pipeline_id} created.")


def create_vector_index(dataset, index_name):
    metadata = []
    content = []

    dataset_docs = json.loads(dataset.read())

    print("extract the content and metadata from the dataset")
    for docs in dataset_docs:
        try:
            if docs["url"] not in ["https://www.stellantis.com/fr/finance",
                                   "https://www.stellantis.com/en/investors"]:
                content.append(docs["body_content"])
                metadata.append({
                    "url": docs["url"],
                    "title": docs["title"],
                    # "meta_description": docs["meta_description"]
                })
        except Exception as exception:
            print("Error: ", docs["title"], docs["meta_description"])
            print(exception)
            exit(1)

    # create a text splitter to split the documents into passages
    nltk_splitter = NLTKTextSplitter(
        chunk_size=800,
        chunk_overlap=160,
    )
    docs_to_embed = nltk_splitter.create_documents(content, metadatas=metadata)
    print("Creating documents...")
    docs_embed = ElasticsearchStore.from_documents(
        docs_to_embed,
        es_cloud_id=CLOUD_ID,
        index_name=index_name,
        es_user=CLOUD_USERNAME,
        es_password=CLOUD_PASSWORD,
        query_field="text_field",
        vector_query_field="vector_query_field.predicted_value",
        strategy=ElasticsearchStore.ApproxRetrievalStrategy(query_model_id=query_model_id),
        bulk_kwargs={"request_timeout": 300, "chunk_size": 500}
    )


if __name__ == "__main__":
    es = Elasticsearch(
        cloud_id=CLOUD_ID,
        basic_auth=(CLOUD_USERNAME, CLOUD_PASSWORD)
    )
    es.info()
    print("Creating index...")
    create_index(es, INDEX_NAME)
    print("Creating ingest pipeline...")
    create_ingest_pipeline(es, INGEST_PIPELINE_ID)
    print("Creating documents...")
    original_dataset = open(DATASET)
    create_vector_index(original_dataset, INDEX_NAME)
    print("Done.")
