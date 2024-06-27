# service-public-elastic-chatbot
Demo application showing how to build a chatbot experience on top of private data using Elasticsearch as vector search
<img width="1842" alt="image" src="https://github.com/akayaz/stellantis-elastic-chatbot/assets/57957968/69000825-04f1-47a7-8091-6b010bd2664c">

<img width="898" alt="image" src="https://github.com/akayaz/stellantis-elastic-chatbot/assets/57957968/eca8742c-dadf-4f7d-b434-d1166fc4570b">



# How to run the application

## Installing and connecting to Elasticsearch
### Install Elasticsearch
There are a number of ways to install Elasticsearch. Cloud is best for most use-cases. Visit the Install Elasticsearch for more information.

### Connect to Elasticsearch
This application requires environment variables to be set. 
Update the .env template provided file with your elastic deployment information
```
CLOUD_ID = elastic_cloud_deployment_id
CLOUD_USERNAME = username
CLOUD_PASSWORD = user_password
ES_VECTOR_INDEX = name_of_the_target_es_vector_index
ELSER_MODEL_ID = id_of_model
DATASET_PATH = path_to_json_file_to_be_indexed
```

## Connecting to LLM
We support multiple LLM providers; By default the application can be configured with 3 providers: Azure, Bedrock and Ollama

### Azure OpenAI
To use Azure openAI, you will to provide to set the following environment variables in the .env file
```
AZURE_OPENAI_API_KEY=''
AZURE_OPENAI_ENDPOINT=''
AZURE_OPENAI_ENGINE=''
AZURE_OPENAI_API_TYPE=azure
AZURE_OPENAI_API_VERSION=''
```
### Bedrock
Follow the instructions on https://docs.aws.amazon.com/bedrock/latest/userguide/api-setup.html to install the python client
by default the application will look for 'anthropic.claude-v2' model

### Ollama
Follow the instructions on https://ollama.com/download/linux to download and install Ollama on your system
Then run the following command to install Mistral
```
ollama run mistral
```

## Activating APM
The application is instrumented using Elastic APM Python agent. 
In order to send the traces to your Elastic Backend and analyze them in the APM application set the following variables in the .env file
```
ELASTIC_APM_SERVER_URL='es-apm-endpoint'
ELASTIC_APM_SECRET_TOKEN='apm_secret_token'
ELASTIC_APM_SERVICE_NAME='stellantis-rag-app'
```

## Run the Application
### Pre-requisites
- Python 3.8+

Install the requirements
```
# create a virtual environment
python -m venv .venv

# activate the virtual environment
source .venv/bin/activate

# Install python dependencies
pip install -r requirements.txt
```

### Ingest data
run the script ``` indexer.py ```

### Launch streamlit application
```
streamlit run chatbot.py --server.port 8501
```

## Screenshots

<img width="1777" alt="image" src="https://github.com/akayaz/stellantis-elastic-chatbot/assets/57957968/c2b784e9-9cf5-4594-ac38-17d1d120c80b">

<img width="1772" alt="image" src="https://github.com/akayaz/stellantis-elastic-chatbot/assets/57957968/dbb3ee4d-205c-4fac-9723-60f24e6c703b">



 
