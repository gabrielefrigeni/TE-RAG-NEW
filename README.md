# RAG Chatbot for Data Governance

## Project Structure

A simplified tree representation of the project folder is:

```bash
.
├── app
│   ├── main.py # Application entry point
│   ├── ingest.py # Ingestion script
│   ├── chat_engine # Main LlamaIndex component
│   |   ├── GeneralInteractionQE # Auxiliary chat tools
│   |   ├── LogHandler # Logger
│   |   ├── SemanticSearchQE # Primary tool for semantic search
│   |   ├── LoadIndex # Vector db loading scripts
│   |   └── load_chat_engine.py # Chat engine definition using the tools
│   ├── utils # Utilities for processing user inputs and chat outputs
│   |   └── user_output.py
│   ├── common # Templates for prompts
│   |   ├── prompt_templates 
│   |   └── base_logger.py
│   ├── mappings_search # Lookup functions to find mappings
│   |   ├── mappings_search.py
│   |   └── response_schema.py
│   └── models # Vertex client and models definitions
│       ├── gcp_client.py
│       └── models.py
└── README.md
```
Check the README files inside each folder for additional information about the files.


## Set Up
Follow these steps to reproduce:

### Installation
- Install the Poetry package manager from: [Poetry Installation Guide](https://python-poetry.org/docs/#installing-with-the-official-installer).

Then, follow these steps to correctly set up the repo:
1. Add to your PATH environment variable: `C:\Users\YOURUSERNAME\AppData\Roaming\Python\Scripts`.
2. Restart your terminal.
3. Create a new conda environment and activate it:
```bash
conda create --name <your-env-name> python=3.12
conda activate <your-env-name>
```
4. Clone this git repository using the url and navigate to the project folder.
5. Locate the Python executable path of the created conda environment using `conda env list` and appending to it `\python.exe`.
6. Run the following command to tell Poetry to use your environment:
```bash
poetry env use <location-of-your-Python-executable>
```
7. Run `poetry check` and then `poetry install` (the check should return the message `All set!`).
8. Set up the GCP credentials in the .env file

**NOTE**: If you introduce new libraries in the code, you must add them to the `pyproject.toml` file. You can do that either manually, by specifying them in the `tool.poetry.dependencies` of the file, or running the command `poetry add <new-library-name>` (recommended approach).

## 4. Running the UI
First, make sure you have a local copy of the Chroma vector database.
In case you don't have it, run the ingestion by:
```bash
poetry run python app/ingest.py
```
This should take just a few seconds.

Now you are ready to go! Run the UI using the command:
```bash
poetry run chainlit run app/main.py
```