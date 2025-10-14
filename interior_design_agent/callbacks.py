from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse, LlmRequest
from google.genai.types import Part


async def before_model_modifier(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> LlmResponse | None:
    for content in llm_request.contents:
        if not content.parts:
            continue

        modified_parts = []
        for i, part in enumerate(content.parts):
            if (
                part.function_response
                and part.function_response.name
                in ["generate_concept_image", "edit_image"]
                and part.function_response.response["generated_image_artifact_id"]
            ):
                artifact_id = part.function_response.response[
                    "generated_image_artifact_id"
                ]
                artifact = await callback_context.load_artifact(filename=artifact_id)

                # Original function response
                modified_parts.append(part)

                # Artifact content appended after function response
                modified_parts.append(
                    Part(text=(f"Below is the content of artifact ID : {artifact_id}"))
                )
                modified_parts.append(artifact)
            else:
                modified_parts.append(part)

        content.parts = modified_parts
