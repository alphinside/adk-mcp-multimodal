# Multimodal Handling with ADK - Artifact Service Utilization

This project is a playground for experimenting with the Google ADK (Agent Development Kit) which emphasize on handling multimodal data. It features a multimodal agent that can handle file and image uploads and process them using custom tools.

## Features

*   **Multimodal Agent:** A sophisticated agent capable of handling various types of user inputs, including text and files.
*   **File Artifacts:** All uploaded files are saved as artifacts, which can be referenced and processed in subsequent interactions.
*   **Image Processing:** Includes a tool to convert images to grayscale, showcasing the integration of custom tools with the agent.
*   **Extensible:** The project is designed to be easily extended with new tools and capabilities.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/adk-playground.git
    cd adk-playground
    ```

2.  **Install dependencies:**
    This project uses `uv` for package management. To install the dependencies, run:
    ```bash
    uv pip install -r requirements.txt
    ```
    If you don't have a `requirements.txt` file, you can generate one from the `pyproject.toml` file:
    ```bash
    uv pip freeze > requirements.txt
    ```

## Usage

To run the agent, you will need to use the `adk` command-line tool.

1.  **Set up your environment:**
    Make sure you have your Google API key and other necessary environment variables set up in a `.env` file.

2.  **Run the agent:**
    ```bash
    adk run multimodal_agent/agent.py
    ```

## Dependencies

This project relies on the following main dependencies:

*   [google-adk](https://github.com/google/agent-development-kit): The core framework for building the agent.
*   [opencv-python](https://pypi.org/project/opencv-python/): A library for computer vision tasks, used here for image processing.

For a full list of dependencies, please see the `pyproject.toml` file.

## Agent Details

### MultimodalAgent

*   **Model:** `gemini-2.5-flash-lite`
*   **Instruction:** "Assist user with their requests"
*   **Description:** The agent is an expert at handling multimodal requests like file or images. Each uploaded file will be saved as an artifact, and the agent will be provided with the artifact information (filename) directly after the file.

### Tools

*   **`convert_image_to_grayscale(artifact_filename: str)`:**
    *   **Description:** Converts an image artifact to black and white (grayscale). This tool loads an image artifact, converts it to grayscale using OpenCV, and saves the processed image as a new artifact.
    *   **Arguments:**
        *   `artifact_filename` (str): The name of the image artifact file to convert.
    *   **Returns:** A dictionary containing the conversion status and metadata.

### Callbacks

*   **`before_agent_callback(callback_context: CallbackContext)`:**
    *   **Description:** This function is called before the agent processes the user's request. It intercepts any uploaded files, saves them as artifacts, and then appends the artifact information to the user's message. This ensures that the agent is aware of the uploaded files and can process them accordingly.
