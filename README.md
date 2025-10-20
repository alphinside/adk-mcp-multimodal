# ADK with Multimodal Tool Interaction

> **⚠️ DISCLAIMER: THIS IS NOT AN OFFICIALLY SUPPORTED GOOGLE PRODUCT. THIS PROJECT IS INTENDED FOR DEMONSTRATION PURPOSES ONLY. IT IS NOT INTENDED FOR USE IN A PRODUCTION ENVIRONMENT.**

This demo showcases how to implement **multimodal tool interaction flow in ADK** using a creative product designer agent use case. In this use case, the agent can refer to the user-uploaded images and perform the required edits by referencing the artifact identifier which is given as context in the model callback. Furthermore, the tool also can produce multimodal data and save it as artifact to be used in the conversation context.

## Prerequisites

- If you are executing this project from your local IDE, Login to Gcloud using CLI with the following command :

    ```shell
    gcloud auth application-default login
    ```

- Enable the following APIs

    ```shell
    gcloud services enable aiplatform.googleapis.com 
    ```

- Install [uv](https://docs.astral.sh/uv/getting-started/installation/) dependencies and prepare the python env

    ```shell
    curl -LsSf https://astral.sh/uv/install.sh | sh
    uv python install 3.12
    uv sync --frozen
    ```

## How to Run

- Copy the `creative_product_designer_agent/.env.example` file to `creative_product_designer_agent/.env` and fill in the values

- Run the agent using the following command:

    ```shell
    uv run adk web
    ```
