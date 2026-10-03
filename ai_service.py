from __future__ import annotations

import base64
import json
import os
import random
import time
from typing import Any

import requests
from google import genai
from google.genai import types
from pydantic import BaseModel, ConfigDict, Field, ValidationError




TEXT_MODEL_DEFAULT = "gemini-3.5-flash-lite"

# Kept for compatibility with main.py.
# ComicCraft no longer uses Gemini for image generation.
IMAGE_MODEL_DEFAULT = "gemini-3.1-flash-image"

STABILITY_IMAGE_URL = (
    "https://api.stability.ai/v2beta/stable-image/generate/core"
)

MAX_IMAGE_BYTES = 8 * 1024 * 1024

MAX_RETRIES = 3
RETRY_DELAYS = (3, 7)



class ComicScene(BaseModel):
    model_config = ConfigDict(extra="ignore")

    scene_description: str = Field(
        ...,
        description="What visually happens in this comic panel.",
    )

    caption: str = Field(
        ...,
        description="Short narration/caption for the panel.",
    )

    dialogue: str = Field(
        ...,
        description="Dialogue spoken in the panel.",
    )

    image_prompt: str = Field(
        ...,
        description="Detailed visual prompt for generating this panel image.",
    )


class ComicStory(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str = Field(
        ...,
        description="Title of the comic.",
    )

    character_description: str = Field(
        ...,
        description="Consistent visual description of the main character.",
    )

    story_text: str = Field(
        ...,
        description="Short complete story summary.",
    )

    scenes: list[ComicScene] = Field(
        ...,
        description="Comic panels/scenes.",
    )




class GeminiServiceError(Exception):
    def __init__(
        self,
        message: str,
        status_code: int | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code

    def __str__(self) -> str:
        return self.message




_client: genai.Client | None = None


def _get_client(api_key: str) -> genai.Client:
    global _client

    api_key = (api_key or "").strip()

    if not api_key:
        raise GeminiServiceError(
            "Gemini API key is missing. "
            "Add GEMINI_API_KEY to the project-root .env file."
        )

    if _client is None:
        _client = genai.Client(api_key=api_key)

    return _client




def _get_error_status(error: Exception) -> int | None:
    """
    Try to extract an HTTP-style status code from an API error.
    """

    for attribute in ("status_code", "code"):
        value = getattr(error, attribute, None)

        if isinstance(value, int):
            return value

        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                pass

    response = getattr(error, "response", None)

    if response is not None:
        value = getattr(response, "status_code", None)

        if isinstance(value, int):
            return value

    return None


def _is_retryable_error(error: Exception) -> bool:
    """
    Retry temporary API failures.
    """

    status = _get_error_status(error)

    if status in {408, 409, 429, 500, 502, 503, 504}:
        return True

    error_text = str(error).lower()

    retry_words = (
        "temporarily unavailable",
        "service unavailable",
        "unavailable",
        "overloaded",
        "high demand",
        "rate limit",
        "too many requests",
        "timeout",
        "timed out",
        "deadline exceeded",
        "internal server error",
    )

    return any(word in error_text for word in retry_words)


def _retry_delay(attempt: int) -> float:
    """
    Return a small retry delay.
    """

    if attempt <= len(RETRY_DELAYS):
        base_delay = RETRY_DELAYS[attempt - 1]
    else:
        base_delay = RETRY_DELAYS[-1]

    return base_delay + random.uniform(0.0, 1.0)


def _friendly_api_error(error: Exception) -> str:
    """
    Convert common Gemini errors into user-friendly messages.
    """

    status = _get_error_status(error)
    error_text = str(error)

    if status in {401, 403}:
        return (
            "Gemini rejected the GEMINI_API_KEY. "
            "Check that the key is valid and enabled for the Gemini API."
        )

    if status == 404:
        return (
            "The selected Gemini model was not found or is not available "
            "for this API key. Check GEMINI_TEXT_MODEL in the .env file."
        )

    if status == 429:
        return (
            "Gemini is temporarily rate-limiting the story request. "
            "Please try again."
        )

    if status in {500, 502, 503, 504}:
        return (
            "Gemini is temporarily unavailable or experiencing high demand. "
            "Please try again."
        )

    if "quota" in error_text.lower():
        return (
            "The Gemini API quota appears to have been reached. "
            "Check the Gemini API project and quota."
        )

    return (
        "ComicCraft could not complete the Gemini story request. "
        "Please try again."
    )




def _extract_json_text(text: str) -> str:
    """
    Remove Markdown code fences if Gemini returns JSON inside
    ```json ... ```
    """

    text = (text or "").strip()

    if text.startswith("```"):
        lines = text.splitlines()

        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    return text


def _read_story_response(
    response: Any,
    panel_count: int,
) -> ComicStory:
    """
    Convert Gemini's structured JSON response into ComicStory.
    """

    

    parsed = getattr(response, "parsed", None)

    if parsed is not None:
        try:
            if isinstance(parsed, ComicStory):
                story = parsed
            else:
                story = ComicStory.model_validate(parsed)

        except ValidationError as error:
            raise GeminiServiceError(
                "Gemini returned structured data that ComicCraft "
                f"could not validate: {error}"
            ) from error

    else:
        

        text = getattr(response, "text", None)

        if not text:
            raise GeminiServiceError(
                "Gemini returned an empty story response."
            )

        text = _extract_json_text(text)

        try:
            data = json.loads(text)

        except json.JSONDecodeError as error:
            raise GeminiServiceError(
                "Gemini returned invalid JSON for the comic story."
            ) from error

        try:
            story = ComicStory.model_validate(data)

        except ValidationError as error:
            raise GeminiServiceError(
                "Gemini returned story data with an invalid structure: "
                f"{error}"
            ) from error

    

    if len(story.scenes) != panel_count:
        raise GeminiServiceError(
            f"Gemini returned {len(story.scenes)} panels, "
            f"but ComicCraft requested {panel_count} panels."
        )

    

    if not story.title.strip():
        raise GeminiServiceError(
            "Gemini returned a story without a title."
        )

    if not story.character_description.strip():
        raise GeminiServiceError(
            "Gemini returned a story without a character description."
        )

    if not story.story_text.strip():
        raise GeminiServiceError(
            "Gemini returned an empty story."
        )

    return story




def generate_story(
    api_key: str,
    model_name: str,
    values: dict[str, str],
) -> ComicStory:
    """
    Generate the complete comic story using Gemini.
    """

    client = _get_client(api_key)

    model_name = (
        model_name or TEXT_MODEL_DEFAULT
    ).strip()

    title = (values.get("title") or "").strip()
    character = (values.get("character") or "").strip()
    genre = (values.get("genre") or "Fantasy").strip()
    description = (values.get("description") or "").strip()

    character_details = (
        values.get("character_details") or ""
    ).strip()

    setting = (values.get("setting") or "").strip()
    tone = (values.get("tone") or "Inspiring").strip()
    art_style = (values.get("art_style") or "Anime").strip()

    try:
        panel_count = int(
            values.get("panel_count") or 4
        )

    except ValueError:
        panel_count = 4

    panel_count = max(2, min(panel_count, 6))

    

    prompt = f"""
You are the story-writing engine for ComicCraft, an AI comic story creator.

Create an original comic story using the user's information below.

USER INPUT
----------
Title idea: {title}
Main character: {character}
Genre: {genre}
Tone: {tone}
Art style: {art_style}
Setting: {setting}
Character description: {character_details}

Story idea:
{description}

REQUIREMENTS
------------
1. Create exactly {panel_count} comic panels.
2. Keep the main character visually consistent across every panel.
3. Make the story have a clear beginning, middle, turning point, and ending.
4. Each panel must move the story forward.
5. Keep dialogue short and natural.
6. Create a useful image_prompt for every panel.
7. The image_prompt must describe:
   - characters
   - clothing/appearance
   - environment
   - action
   - camera framing
   - lighting
   - mood
   - comic art style
8. Do not put text, captions, speech bubbles, or UI elements
   inside the generated artwork.
9. The character_description should be detailed enough to keep
   the character consistent between generated images.
10. Return only the requested structured story data.

The generated story must contain:
- title
- character_description
- story_text
- scenes

Every scene must contain:
- scene_description
- caption
- dialogue
- image_prompt
"""

    

    for attempt in range(1, MAX_RETRIES + 1):

        print(
            f"[ComicCraft] Gemini story request "
            f"attempt {attempt}/{MAX_RETRIES}..."
        )

        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ComicStory,
                    max_output_tokens=2400,
                ),
            )

            print("[ComicCraft] ✓ Story response received.")

            story = _read_story_response(
                response,
                panel_count,
            )

            if title:
                story.title = title

            print(
                "[ComicCraft] ✓ Story generated successfully."
            )

            return story

        except ValidationError as error:

            print(
                "[ComicCraft] FULL STORY ERROR TYPE:",
                type(error).__name__,
            )

            print(
                "[ComicCraft] FULL STORY ERROR:",
                repr(error),
            )

            raise GeminiServiceError(
                "ComicCraft received story data in an invalid format."
            ) from error

        except Exception as error:

            print(
                "[ComicCraft] FULL STORY ERROR TYPE:",
                type(error).__name__,
            )

            print(
                "[ComicCraft] FULL STORY ERROR:",
                repr(error),
            )

            if (
                attempt < MAX_RETRIES
                and _is_retryable_error(error)
            ):
                delay = _retry_delay(attempt)

                print(
                    "[ComicCraft] Temporary Gemini story error. "
                    f"Retrying in {delay:.1f} seconds..."
                )

                time.sleep(delay)
                continue

            raise GeminiServiceError(
                _friendly_api_error(error),
                _get_error_status(error),
            ) from error

    raise GeminiServiceError(
        "ComicCraft could not generate the story."
    )




def _get_stability_api_key() -> str:
    """
    Read Stability API key from the environment.

    main.py already loads the project .env file before importing
    ai_service.py.
    """

    api_key = os.getenv(
        "STABILITY_API_KEY",
        "",
    ).strip()

    if not api_key:
        raise GeminiServiceError(
            "Stability AI API key is missing. "
            "Add STABILITY_API_KEY to the project-root .env file."
        )

    return api_key


def _stability_error_message(
    response: requests.Response,
) -> str:
    """
    Convert Stability API errors into a readable message.
    """

    status = response.status_code

    try:
        payload = response.json()

        errors = payload.get("errors")

        if isinstance(errors, list) and errors:
            error_text = "; ".join(
                str(item)
                for item in errors
            )
        else:
            error_text = str(payload)

    except Exception:
        error_text = response.text.strip()

    if status == 401:
        return (
            "Stability AI rejected the API key. "
            "Check STABILITY_API_KEY in the .env file."
        )

    if status == 403:
        return (
            "Stability AI rejected the request. "
            "Check your Stability account permissions or credits."
        )

    if status == 400:
        return (
            "Stability AI rejected the image request: "
            f"{error_text}"
        )

    if status == 422:
        return (
            "Stability AI could not process the image request: "
            f"{error_text}"
        )

    if status == 429:
        return (
            "Stability AI is rate-limiting the image request. "
            "Please wait and try again."
        )

    if status >= 500:
        return (
            "Stability AI is temporarily unavailable. "
            "Please try again."
        )

    return (
        f"Stability AI returned HTTP {status}: "
        f"{error_text}"
    )


def _is_retryable_stability_status(
    status_code: int,
) -> bool:
    """
    Return True for temporary Stability API failures.
    """

    return status_code in {
        408,
        429,
        500,
        502,
        503,
        504,
    }



def generate_panel_image(
    api_key: str,
    model_name: str,
    story: ComicStory,
    panel_index: int,
    reference_image: bytes | None = None,
    reference_mime_type: str = "image/png",
) -> tuple[bytes, str]:
    """
    Generate one comic panel using Stability AI Stable Image Core.

    The api_key and model_name parameters are kept in the function
    signature so main.py does not need to change.

    Stability AI reads its own key from:
        STABILITY_API_KEY
    """

    del api_key
    del model_name

    if not 0 <= panel_index < len(story.scenes):
        raise GeminiServiceError(
            "Invalid comic panel number."
        )

    stability_api_key = _get_stability_api_key()

    scene = story.scenes[panel_index]

    

    character_description = (
        story.character_description.strip()
    )

    scene_description = (
        scene.scene_description.strip()
    )

    image_prompt = (
        scene.image_prompt.strip()
    )

    dialogue = (
        scene.dialogue.strip()
    )

    

    prompt = f"""
Create a single finished comic-book panel illustration.

MAIN CHARACTER
--------------
{character_description}

CURRENT SCENE
-------------
{scene_description}

DETAILED VISUAL DIRECTION
-------------------------
{image_prompt}

DIALOGUE CONTEXT
----------------
{dialogue}

VISUAL REQUIREMENTS
-------------------
- Professional comic-book illustration.
- Consistent main character appearance.
- Keep face, hairstyle, clothing, body proportions,
  colors, and important visual traits consistent.
- Clear storytelling composition.
- Strong foreground, middle ground, and background.
- Expressive character pose and facial expression.
- Detailed environment.
- Cinematic lighting.
- Appropriate mood for the scene.
- Landscape comic panel composition.
- Clean line art.
- High-quality finished artwork.
- No speech bubbles.
- No captions.
- No UI.
- No written words inside the artwork.
- No watermark.
"""

    negative_prompt = (
        "text, words, letters, captions, subtitles, speech bubbles, "
        "watermark, logo, UI, interface, distorted face, "
        "extra fingers, extra limbs, duplicate character, "
        "blurry image, low quality"
    )

    

    if reference_image:
        print(
            "[ComicCraft] Character reference available; "
            "using the detailed character description for "
            "Stable Image Core generation."
        )

    
    headers = {
        "authorization": (
            f"Bearer {stability_api_key}"
        ),
        "accept": "image/*",
    }

    data = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "aspect_ratio": "3:2",
        "style_preset": "comic-book",
        "output_format": "png",
    }

    

    for attempt in range(1, MAX_RETRIES + 1):

        print(
            f"[ComicCraft] Stability image request "
            f"for panel {panel_index + 1} "
            f"attempt {attempt}/{MAX_RETRIES}..."
        )

        try:
            response = requests.post(
                STABILITY_IMAGE_URL,
                headers=headers,
                files={
                    "none": "",
                },
                data=data,
                timeout=180,
            )

        except requests.RequestException as error:

            print(
                "[ComicCraft] FULL IMAGE ERROR TYPE:",
                type(error).__name__,
            )

            print(
                "[ComicCraft] FULL IMAGE ERROR:",
                repr(error),
            )

            if attempt < MAX_RETRIES:

                delay = _retry_delay(attempt)

                print(
                    "[ComicCraft] Stability network error. "
                    f"Retrying in {delay:.1f} seconds..."
                )

                time.sleep(delay)
                continue

            raise GeminiServiceError(
                "Could not connect to Stability AI. "
                "Please check your internet connection."
            ) from error

        

        if response.status_code == 200:

            image_bytes = response.content

            if not image_bytes:
                raise GeminiServiceError(
                    "Stability AI returned an empty image."
                )

            if len(image_bytes) > MAX_IMAGE_BYTES:
                raise GeminiServiceError(
                    "Stability AI returned an image larger "
                    "than ComicCraft allows."
                )

            print(
                f"[ComicCraft] ✓ Stability image response "
                f"received for panel {panel_index + 1}."
            )

            print(
                f"[ComicCraft] ✓ Panel {panel_index + 1} image "
                f"generated successfully "
                f"({len(image_bytes):,} bytes)."
            )

            return image_bytes, "image/png"

        

        print(
            "[ComicCraft] FULL IMAGE ERROR TYPE:",
            "StabilityAPIError",
        )

        print(
            "[ComicCraft] FULL IMAGE ERROR:",
            response.text[:2000],
        )

        status = response.status_code

        # Retry only temporary errors.
        if (
            attempt < MAX_RETRIES
            and _is_retryable_stability_status(status)
        ):

            delay = _retry_delay(attempt)

            print(
                "[ComicCraft] Temporary Stability image error. "
                f"Retrying in {delay:.1f} seconds..."
            )

            time.sleep(delay)
            continue

        raise GeminiServiceError(
            _stability_error_message(response),
            status,
        )

    raise GeminiServiceError(
        f"ComicCraft could not generate panel "
        f"{panel_index + 1}."
    )