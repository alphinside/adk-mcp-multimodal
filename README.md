# ADK with Multimodal Tool Interaction

> **⚠️ DISCLAIMER: THIS IS NOT AN OFFICIALLY SUPPORTED GOOGLE PRODUCT. THIS PROJECT IS INTENDED FOR DEMONSTRATION PURPOSES ONLY. IT IS NOT INTENDED FOR USE IN A PRODUCTION ENVIRONMENT.**

This demo showcases how to implement **multimodal tool interaction flow in ADK** using a creative product marketing agent use case. In this use case, the agent can refer to the user-uploaded images and perform the required edits by referencing the artifact identifier which is given as context in the model callback. Furthermore, the tool can produce multimodal data (images and videos) and save it as artifact to be used in the conversation context.

**Key Capabilities:**

- **Image Editing**: Transform and edit product photos using custom tools
- **Video Generation**: Create professional product marketing video clips from images using Google's Veo 3.1 API
- **Multimodal Context**: Seamlessly reference and work with both image and video artifacts throughout the conversation
- **Automatic Prompt Enrichment**: User prompts are automatically enhanced with professional production quality guidelines for marketing-ready videos

![Photo Editing](assets/photo_editing.gif)

![Video Generation](assets/video_creation.gif)

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

1. Rename `example_full_agent` dir to `product_photo_editor`

2. Copy the `product_photo_editor/.env.example` file to `product_photo_editor/.env` and fill in the values

3. Rename `example_mcp_server` dir to `veo_mcp`

4. Copy the `veo_mcp/.env.example` file to `veo_mcp/.env` and fill in the values

5. Run the agent using the following command:

    ```shell
    uv run adk web
    ```
