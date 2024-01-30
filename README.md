# Tools and prerequisites

To build your LLM and set up automated testing, you’ll need the following frameworks and tools:

- [CircleCI](https://circleci.com/signup/) – A configuration-as-code CI/CD platform that allows users to run and
  orchestrate cloud-based pipelines across machines.

- [LangChain](https://docs.langchain.com/docs/) – An open-source framework for developing language model-powered
  applications. It provides prompt templates, models, document loaders, text splitters,
  and [many other tools for interacting with models](https://docs.langchain.com/docs/category/components).

- [PareaAI](https://www.parea.ai/) – A tool created to more efficiently debug language model applications
  by showing the trace of LLM calls, as well as inputs and outputs for certain prompts. This allows you to view the test
  results and metadata for all LLM calls in a single dashboard.

- [ChromaDB](https://docs.trychroma.com/) – An open-source embedding database/vector store, which tokenizes inputs (in
  our case, text) and stores them in an n-dimensional vector space. Chunks of text similar to new inputs can be returned
  using a modified K-nearest-neighbor algorithm in the vector space.

# Directory structure

- The rag/ directory contains an example LLM-powered application and unit test suite spread across multiple Python
  scripts.
    - These scripts rely on a .env file with API keys to OpenAI and Langchain, as well as other environment variables.
      An example is provided, but you need to populate it with your own variables.
- .circleci/ directory contains the CircleCI config.yml that defines the CircleCI pipelines that call the
  ML scripts.

# Set up environment

- Fork the example repository and clone your forked version locally.

- Install and activate your virtual environment

- Create a .env file by running `cp .env.example .env`, then set the necessary environment variables in it. This will
  include:
    - **OpenAI API key**: Go
      to [https://platform.openai.com/account/api-keys](https://platform.openai.com/account/api-keys), set up a paid
      account, and create a new secret key. This key should be stored in the `OPENAI_API_KEY` environment variable in
      your .env file.
    - **Parea API key**: Go
      to [https://app.parea.ai/settings](https://docs.parea.ai/api-reference/authentication#parea-api-key), create an
      account, and create an API key by clicking on the API Keys button on the bottom left of the page and following the
      instructions. This key should be stored in the `PAREA_API_KEY` environment variable in your .env file.

# Run the application

## Start the server

    flask --app rag/app run &  # spin up the Flask server

## Interact with the server

    curl -X POST -H "Content-Type: application/json" -d '{"message": "What is LangSmith?"}' http://127.0.0.1:5000

Ask the application different questions by changing the `message` field in the curl command.

## Stop the server

    fg [job no. of server]
    Ctrl-C