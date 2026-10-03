from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import time
import uuid
from html import escape
from io import BytesIO
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, Form, Request
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ValidationError
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image as PdfImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

import ai_service


# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_FILE, override=False)

app = FastAPI(title="ComicCraft - AI Comic Story Creator")

app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "static"),
    name="static",
)

templates = Jinja2Templates(directory=BASE_DIR / "templates")

GENERATION_DIR = BASE_DIR / ".comiccraft" / "generations"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

GEMINI_TEXT_MODEL = (
    os.getenv("GEMINI_TEXT_MODEL")
    or ai_service.TEXT_MODEL_DEFAULT
).strip()

GEMINI_IMAGE_MODEL = (
    os.getenv("GEMINI_IMAGE_MODEL")
    or ai_service.IMAGE_MODEL_DEFAULT
).strip()


# Temporary Gemini configuration check
print("[ComicCraft] API key loaded:", bool(GEMINI_API_KEY))
print("[ComicCraft] API key length:", len(GEMINI_API_KEY))
print("[ComicCraft] Text model:", GEMINI_TEXT_MODEL)
print("[ComicCraft] Image model:", GEMINI_IMAGE_MODEL)
# ============================================================
# GENERATION SETTINGS
# ============================================================

GENERATION_TTL_SECONDS = 12 * 60 * 60
MAX_SAVED_GENERATIONS = 8

# Only one full comic generation at a time.
MAX_ACTIVE_GENERATIONS = 1

# Prevent accidental double-click / repeated generation.
GENERATION_COOLDOWN_SECONDS = 30

# Panel regeneration has a shorter cooldown.
IMAGE_REGENERATION_COOLDOWN_SECONDS = 5

_active_generations = 0

_last_generation_by_client: dict[str, float] = {}
_last_image_generation_by_client: dict[str, float] = {}

_generation_cache: dict[str, StoredGeneration] = {}


# ============================================================
# DATA MODELS
# ============================================================

class StoredGeneration(BaseModel):
    generation_id: str
    values: dict[str, str]
    story: ai_service.ComicStory
    image_mime_types: list[str]
    created_at: float
    is_demo: bool = False


# ============================================================
# DEFAULT / UI DATA
# ============================================================

DEMO_STORY = {
    "title": "The Lantern at Low Tide",
    "character": "Mira",
    "genre": "Fantasy",
    "description": (
        "Mira finds a tiny lantern glowing beneath the old pier. "
        "A shy sea dragon appears and asks for help finding its way home. "
        "Together they follow a trail of blue lights across the bay."
    ),
}

EMPTY_STORY = {
    "title": "",
    "character": "",
    "genre": "Fantasy",
    "description": "",
    "character_details": "",
    "setting": "",
    "tone": "Inspiring",
    "art_style": "Anime",
    "panel_count": "4",
}

WORKFLOW_STEPS = {
    "welcome": 1,
    "story": 2,
    "preview": 3,
    "final": 4,
}

GENRE_DIALOGUE = {
    "Fantasy": [
        "That light is calling to us.",
        "Then let's follow it together.",
    ],
    "Sci-Fi": [
        "The signal is coming from inside the nebula.",
        "Plotting a course. Hold on tight.",
    ],
    "Mystery": [
        "The clue was here the whole time.",
        "Then we know where to look next.",
    ],
    "Adventure": [
        "The map ends at the edge of the world.",
        "Good. That's where the fun begins.",
    ],
    "Comedy": [
        "I meant to do that. Mostly.",
        "Sure. Let's call it a plan.",
    ],
    "Romance": [
        "I think this is where our story begins.",
        "Then let's write it together.",
    ],
}

GENRE_CHOICES = tuple(GENRE_DIALOGUE)

TONE_CHOICES = (
    "Funny",
    "Emotional",
    "Dark",
    "Inspiring",
)

ART_STYLE_CHOICES = (
    "Anime",
    "Manga",
    "Cartoon",
    "3D",
    "Watercolor",
)

PANEL_COUNT_CHOICES = (
    "2",
    "3",
    "4",
    "5",
    "6",
)

PANEL_BEATS = (
    "THE SPARK",
    "THE JOURNEY",
    "THE TURN",
    "THE NEW DAY",
    "THE DISCOVERY",
    "THE HORIZON",
)


# ============================================================
# FILE / GENERATION HELPERS
# ============================================================

def _generation_directory(generation_id: str) -> Path | None:
    try:
        parsed_id = uuid.UUID(generation_id)
    except (ValueError, AttributeError, TypeError):
        return None

    if str(parsed_id) != generation_id.lower():
        return None

    return GENERATION_DIR / str(parsed_id)


def _image_extension(mime_type: str) -> str:
    extensions = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
    }

    try:
        return extensions[mime_type.lower()]
    except KeyError as error:
        raise ValueError(
            "Gemini returned an unsupported image format."
        ) from error


def _image_path(
    generation: StoredGeneration,
    panel_index: int,
) -> Path | None:

    if not 0 <= panel_index < len(generation.image_mime_types):
        return None

    directory = _generation_directory(
        generation.generation_id
    )

    if directory is None:
        return None

    try:
        extension = _image_extension(
            generation.image_mime_types[panel_index]
        )
    except ValueError:
        return None

    return directory / f"panel-{panel_index}{extension}"


def _load_generation(
    generation_id: str,
) -> StoredGeneration | None:

    directory = _generation_directory(generation_id)

    if directory is None:
        return None

    cached = _generation_cache.get(generation_id)

    if cached is not None:

        if time.time() - cached.created_at > GENERATION_TTL_SECONDS:
            _generation_cache.pop(generation_id, None)
            shutil.rmtree(directory, ignore_errors=True)
            return None

        if cached.is_demo and not cached.image_mime_types:
            return cached

        if (
            len(cached.image_mime_types)
            == len(cached.story.scenes)
            and all(
                (
                    path := _image_path(cached, index)
                ) is not None
                and path.is_file()
                for index in range(len(cached.story.scenes))
            )
        ):
            return cached

        _generation_cache.pop(generation_id, None)

    manifest = directory / "manifest.json"

    try:
        generation = StoredGeneration.model_validate_json(
            manifest.read_text(encoding="utf-8")
        )

        if generation.generation_id != generation_id:
            return None

        if (
            time.time() - generation.created_at
            > GENERATION_TTL_SECONDS
        ):
            shutil.rmtree(directory, ignore_errors=True)
            return None

        if generation.is_demo and not generation.image_mime_types:
            pass

        elif (
            len(generation.image_mime_types)
            != len(generation.story.scenes)
            or any(
                _image_path(generation, index) is None
                or not _image_path(generation, index).is_file()
                for index in range(len(generation.story.scenes))
            )
        ):
            return None

    except (
        OSError,
        ValueError,
        ValidationError,
    ):
        return None

    _generation_cache[generation_id] = generation

    return generation


def _prune_generations() -> None:

    if not GENERATION_DIR.exists():
        return

    directories = sorted(
        (
            path
            for path in GENERATION_DIR.iterdir()
            if path.is_dir()
        ),
        key=lambda path: path.stat().st_mtime,
    )

    cutoff = time.time() - GENERATION_TTL_SECONDS

    live_directories = []

    for directory in directories:

        if directory.stat().st_mtime < cutoff:

            shutil.rmtree(
                directory,
                ignore_errors=True,
            )

            _generation_cache.pop(
                directory.name,
                None,
            )

        else:
            live_directories.append(directory)

    for directory in live_directories[:-MAX_SAVED_GENERATIONS]:

        shutil.rmtree(
            directory,
            ignore_errors=True,
        )

        _generation_cache.pop(
            directory.name,
            None,
        )


def _save_generation(
    generation_id: str,
    values: dict[str, str],
    story: ai_service.ComicStory,
    images: list[tuple[bytes, str]],
    is_demo: bool = False,
) -> StoredGeneration:

    directory = _generation_directory(
        generation_id
    )

    if (
        directory is None
        or (
            not is_demo
            and len(images) != len(story.scenes)
        )
        or (
            is_demo
            and images
        )
    ):
        raise ValueError(
            "A complete comic generation is required."
        )

    GENERATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    directory.mkdir()

    image_mime_types = [
        mime_type.lower()
        for _, mime_type in images
    ]

    record = StoredGeneration(
        generation_id=generation_id,
        values=values,
        story=story,
        image_mime_types=image_mime_types,
        created_at=time.time(),
        is_demo=is_demo,
    )

    try:

        for index, (
            image_bytes,
            mime_type,
        ) in enumerate(images):

            path = (
                directory
                / f"panel-{index}"
                f"{_image_extension(mime_type)}"
            )

            path.write_bytes(image_bytes)

        manifest_tmp = directory / "manifest.tmp"

        manifest_tmp.write_text(
            record.model_dump_json(),
            encoding="utf-8",
        )

        manifest_tmp.replace(
            directory / "manifest.json"
        )

    except Exception:

        shutil.rmtree(
            directory,
            ignore_errors=True,
        )

        raise

    _generation_cache[generation_id] = record

    _prune_generations()

    return record


# ============================================================
# PANEL / COMIC HELPERS
# ============================================================

def _panel_payload(
    generation: StoredGeneration,
) -> list[dict[str, str]]:

    panels = []

    for index, scene in enumerate(
        generation.story.scenes
    ):

        panel = {
            **scene.model_dump(),
            "beat": PANEL_BEATS[
                index % len(PANEL_BEATS)
            ],
            "art": (
                "lantern",
                "path",
                "spark",
                "horizon",
            )[index % 4],
        }

        if not generation.is_demo:

            panel["image_url"] = (
                f"/generated/"
                f"{generation.generation_id}"
                f"/panel/{index}"
            )

        panels.append(panel)

    return panels


def _shorten(
    text: str,
    limit: int = 22,
) -> str:

    words = text.split()

    if len(words) <= limit:
        return text

    return (
        " ".join(words[:limit])
        .rstrip(".,;:")
        + "..."
    )


def generate_comic(
    title: str,
    character: str,
    genre: str,
    description: str,
) -> list[dict[str, str]]:

    sentences = [
        part.strip()
        for part in re.split(
            r"(?<=[.!?])\s+",
            description.strip(),
        )
        if part.strip()
    ]

    opening = (
        _shorten(sentences[0])
        if sentences
        else "A new idea begins to take shape."
    )

    middle = (
        _shorten(sentences[1])
        if len(sentences) > 1
        else "A surprising obstacle changes the plan."
    )

    turning_point = (
        _shorten(sentences[2])
        if len(sentences) > 2
        else "A small clue reveals a clever way forward."
    )

    ending = (
        _shorten(sentences[-1])
        if len(sentences) > 1
        else "A hard-won discovery opens a new beginning."
    )

    dialogue = GENRE_DIALOGUE.get(
        genre,
        GENRE_DIALOGUE["Fantasy"],
    )

    return [
        {
            "beat": PANEL_BEATS[0],
            "caption": (
                f"{character} steps into a "
                f"{genre.lower()} story: {opening}"
            ),
            "dialogue": (
                f"{character}: {dialogue[0]}"
            ),
            "art": "lantern",
        },
        {
            "beat": PANEL_BEATS[1],
            "caption": (
                "The journey takes an unexpected turn. "
                f"{middle}"
            ),
            "dialogue": (
                f"{character}: {dialogue[1]}"
            ),
            "art": "path",
        },
        {
            "beat": PANEL_BEATS[2],
            "caption": (
                "A new perspective changes everything. "
                f"{turning_point}"
            ),
            "dialogue": (
                f"{character}: "
                "Maybe the answer was closer than I thought."
            ),
            "art": "spark",
        },
        {
            "beat": PANEL_BEATS[3],
            "caption": (
                f"{ending} "
                f"{character} is ready for whatever comes next."
            ),
            "dialogue": (
                f"{character}: This is only the beginning."
            ),
            "art": "horizon",
        },
    ]


# ============================================================
# DEMO GENERATION
# ============================================================

def _generate_demo_generation(
    values: dict[str, str],
) -> StoredGeneration:

    character = values["character"]

    sentences = [
        part.strip()
        for part in re.split(
            r"(?<=[.!?])\s+",
            values["description"],
        )
        if part.strip()
    ]

    panel_count = int(
        values["panel_count"]
    )

    dialogue = GENRE_DIALOGUE[
        values["genre"]
    ]

    scenes = []

    for index in range(panel_count):

        beat = PANEL_BEATS[
            index % len(PANEL_BEATS)
        ]

        detail = sentences[
            min(index, len(sentences) - 1)
        ]

        scenes.append(
            ai_service.ComicScene(
                scene_description=(
                    f"{character} reaches "
                    f"{values.get('setting') or 'a new turn in the adventure'}: "
                    f"{detail}"
                ),
                caption=(
                    f"{beat.title()}: {detail}"
                ),
                dialogue=(
                    f"{character}: "
                    f"{dialogue[index % len(dialogue)]}"
                ),
                image_prompt=(
                    "Demo mode only. "
                    "No AI-generated image is included."
                ),
            )
        )

    setting = (
        values.get("setting")
        or "a world of their own"
    )

    story = ai_service.ComicStory(
        title=values["title"],
        character_description=(
            values["character_details"]
            or "No visual character description was provided."
        ),
        story_text=(
            f"In {setting}, {character} begins "
            f"a {values['genre'].lower()} adventure. "
            f"{values['description']} "
            f"With a little courage and a new perspective, "
            f"{character} takes the next step "
            "and discovers that every great story "
            "can begin with one small idea."
        ),
        scenes=scenes,
    )

    return _save_generation(
        str(uuid.uuid4()),
        values,
        story,
        [],
        is_demo=True,
    )


# ============================================================
# PAGE RENDERING
# ============================================================

def _render_page(
    request: Request,
    values: dict[str, str],
    comic: list[dict[str, str]] | None = None,
    error: str | None = None,
    generated: bool = False,
    status_code: int = 200,
    stage: str = "welcome",
    generated_story: ai_service.ComicStory | None = None,
    generation_id: str | None = None,
    is_demo: bool = False,
) -> HTMLResponse:

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "values": values,
            "genres": GENRE_CHOICES,
            "tones": TONE_CHOICES,
            "art_styles": ART_STYLE_CHOICES,
            "panel_counts": PANEL_COUNT_CHOICES,
            "comic": (
                comic
                if comic is not None
                else []
            ),
            "error": error,
            "generated": generated,
            "generated_story": generated_story,
            "generation_id": generation_id,
            "is_demo": is_demo,
            "api_configured": bool(
                GEMINI_API_KEY
            ),
            "stage": stage,
            "step": WORKFLOW_STEPS[stage],
            "workflow_steps": WORKFLOW_STEPS,
        },
        status_code=status_code,
    )


# ============================================================
# VALIDATION
# ============================================================

def _validate_story(
    title: str,
    character: str,
    genre: str,
    description: str,
    character_details: str = "",
    setting: str = "",
    tone: str = "Inspiring",
    art_style: str = "Anime",
    panel_count: str = "4",
) -> tuple[dict[str, str], str | None]:

    values = {
        "title": title.strip(),
        "character": character.strip(),
        "genre": genre.strip(),
        "description": description.strip(),
        "character_details": character_details.strip(),
        "setting": setting.strip(),
        "tone": tone.strip(),
        "art_style": art_style.strip(),
        "panel_count": panel_count.strip(),
    }

    if (
        not values["title"]
        or not values["character"]
        or not values["description"]
    ):
        return (
            values,
            "Add a title, a character, and a few story details to begin.",
        )

    if (
        len(values["title"]) > 80
        or len(values["character"]) > 48
        or len(values["description"]) > 600
        or len(values["character_details"]) > 400
        or len(values["setting"]) > 160
    ):
        return (
            values,
            "Keep the title under 80 characters, "
            "character under 48, story under 600, "
            "character details under 400, and setting under 160.",
        )

    if values["genre"] not in GENRE_CHOICES:
        return (
            values,
            "Choose one of the available genres and try again.",
        )

    if values["tone"] not in TONE_CHOICES:
        return (
            values,
            "Choose one of the available story tones and try again.",
        )

    if values["art_style"] not in ART_STYLE_CHOICES:
        return (
            values,
            "Choose one of the available comic art styles and try again.",
        )

    if values["panel_count"] not in PANEL_COUNT_CHOICES:
        return (
            values,
            "Choose between 2 and 6 comic panels.",
        )

    return values, None


# ============================================================
# NON-STREAMING COMPLETE GENERATION
# ============================================================

def _generate_complete(
    values: dict[str, str],
) -> StoredGeneration:

    if not GEMINI_API_KEY:
        raise ai_service.GeminiServiceError(
            "Real AI generation needs a Gemini API key. "
            "Add GEMINI_API_KEY to the project-root .env file "
            "and restart ComicCraft.",
            503,
        )

    generation_id = str(uuid.uuid4())

    story = ai_service.generate_story(
        GEMINI_API_KEY,
        GEMINI_TEXT_MODEL,
        values,
    )

    images: list[tuple[bytes, str]] = []

    for index in range(
        len(story.scenes)
    ):

        reference_bytes, reference_mime = (
            images[0]
            if images
            else (None, "image/jpeg")
        )

        images.append(
            ai_service.generate_panel_image(
                GEMINI_API_KEY,
                GEMINI_IMAGE_MODEL,
                story,
                index,
                reference_image=reference_bytes,
                reference_mime_type=reference_mime,
            )
        )

    return _save_generation(
        generation_id,
        values,
        story,
        images,
    )


# ============================================================
# GENERATED PAGE
# ============================================================

def _render_generation(
    request: Request,
    generation: StoredGeneration,
    stage: str,
) -> HTMLResponse:

    return _render_page(
        request,
        generation.values,
        _panel_payload(generation),
        generated=True,
        stage=stage,
        generated_story=generation.story,
        generation_id=generation.generation_id,
        is_demo=generation.is_demo,
    )


# ============================================================
# SERVER-SENT EVENTS
# ============================================================

def _sse_event(
    payload: dict[str, Any],
) -> str:

    return (
        f"data: "
        f"{json.dumps(payload, ensure_ascii=True)}"
        "\n\n"
    )


def _console_progress(
    message: str,
) -> None:

    print(
        f"[ComicCraft] {message}",
        flush=True,
    )


# ============================================================
# GENERATION LIMIT / COOLDOWN
# ============================================================

def _reserve_generation_slot(
    client_id: str,
    image: bool = False,
) -> tuple[str, int] | None:

    global _active_generations

    cooldowns = (
        _last_image_generation_by_client
        if image
        else _last_generation_by_client
    )

    cooldown = (
        IMAGE_REGENERATION_COOLDOWN_SECONDS
        if image
        else GENERATION_COOLDOWN_SECONDS
    )

    now = time.monotonic()

    stale_before = (
        now
        - max(
            GENERATION_COOLDOWN_SECONDS,
            IMAGE_REGENERATION_COOLDOWN_SECONDS,
        )
        * 10
    )

    for stale_client in [
        key
        for key, timestamp in cooldowns.items()
        if timestamp < stale_before
    ]:
        cooldowns.pop(
            stale_client,
            None,
        )

    previous = cooldowns.get(
        client_id,
        0,
    )

    if now - previous < cooldown:
        return (
            "Please wait a few seconds before starting another generation.",
            429,
        )

    if _active_generations >= MAX_ACTIVE_GENERATIONS:
        return (
            "Another comic is being generated right now. "
            "Please try again shortly.",
            429,
        )

    cooldowns[client_id] = now

    while len(cooldowns) > 1024:

        oldest_client = min(
            cooldowns,
            key=cooldowns.get,
        )

        cooldowns.pop(
            oldest_client,
            None,
        )

    _active_generations += 1

    return None


def _release_generation_slot() -> None:

    global _active_generations

    _active_generations = max(
        0,
        _active_generations - 1,
    )


# ============================================================
# HOME
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse,
)
async def home(
    request: Request,
) -> HTMLResponse:

    return _render_page(
        request,
        EMPTY_STORY,
        stage="welcome",
    )


# ============================================================
# WELCOME
# ============================================================

@app.post(
    "/welcome",
    response_class=HTMLResponse,
)
async def welcome_previous(
    request: Request,
    title: str = Form(""),
    character: str = Form(""),
    genre: str = Form("Fantasy"),
    description: str = Form(""),
    character_details: str = Form(""),
    setting: str = Form(""),
    tone: str = Form("Inspiring"),
    art_style: str = Form("Anime"),
    panel_count: str = Form("4"),
) -> HTMLResponse:

    values, _ = _validate_story(
        title,
        character,
        genre,
        description,
        character_details,
        setting,
        tone,
        art_style,
        panel_count,
    )

    return _render_page(
        request,
        values,
        stage="welcome",
    )


# ============================================================
# STORY PAGE
# ============================================================

@app.get(
    "/story",
    response_class=HTMLResponse,
)
async def story_page(
    request: Request,
    new: bool = False,
) -> HTMLResponse:

    return _render_page(
        request,
        EMPTY_STORY,
        stage="story",
    )


@app.post(
    "/story",
    response_class=HTMLResponse,
)
async def story_from_welcome(
    request: Request,
    title: str = Form(""),
    character: str = Form(""),
    genre: str = Form("Fantasy"),
    description: str = Form(""),
    character_details: str = Form(""),
    setting: str = Form(""),
    tone: str = Form("Inspiring"),
    art_style: str = Form("Anime"),
    panel_count: str = Form("4"),
) -> HTMLResponse:

    values, _ = _validate_story(
        title,
        character,
        genre,
        description,
        character_details,
        setting,
        tone,
        art_style,
        panel_count,
    )

    return _render_page(
        request,
        values,
        stage="story",
    )


@app.post(
    "/edit",
    response_class=HTMLResponse,
)
async def edit_story(
    request: Request,
    title: str = Form(""),
    character: str = Form(""),
    genre: str = Form("Fantasy"),
    description: str = Form(""),
    character_details: str = Form(""),
    setting: str = Form(""),
    tone: str = Form("Inspiring"),
    art_style: str = Form("Anime"),
    panel_count: str = Form("4"),
) -> HTMLResponse:

    values, _ = _validate_story(
        title,
        character,
        genre,
        description,
        character_details,
        setting,
        tone,
        art_style,
        panel_count,
    )

    return _render_page(
        request,
        values,
        stage="story",
    )


# ============================================================
# PREVIEW
# ============================================================

@app.get(
    "/preview",
    response_class=HTMLResponse,
)
async def preview_generated(
    request: Request,
    generation_id: str = "",
) -> Response:

    generation = _load_generation(
        generation_id
    )

    if generation is None:
        return RedirectResponse(
            "/story",
            status_code=303,
        )

    return _render_generation(
        request,
        generation,
        "preview",
    )


# ============================================================
# STREAMING AI GENERATION
# ============================================================

@app.post("/api/generate")
async def stream_generation(
    request: Request,
    title: str = Form(""),
    character: str = Form(""),
    genre: str = Form("Fantasy"),
    description: str = Form(""),
    character_details: str = Form(""),
    setting: str = Form(""),
    tone: str = Form("Inspiring"),
    art_style: str = Form("Anime"),
    panel_count: str = Form("4"),
    generation_mode: str = Form("gemini"),
) -> Response:

    # --------------------------------------------------------
    # Validate user input
    # --------------------------------------------------------

    values, error = _validate_story(
        title,
        character,
        genre,
        description,
        character_details,
        setting,
        tone,
        art_style,
        panel_count,
    )

    if error:
        return JSONResponse(
            {"message": error},
            status_code=422,
        )

    if generation_mode not in {
        "gemini",
        "demo",
    }:
        return JSONResponse(
            {
                "message": (
                    "Choose Gemini or demo generation."
                )
            },
            status_code=422,
        )

    # --------------------------------------------------------
    # Demo mode
    # --------------------------------------------------------

    use_demo = (
        generation_mode == "demo"
        or not GEMINI_API_KEY
    )

    if use_demo:

        try:

            generation = await asyncio.to_thread(
                _generate_demo_generation,
                values,
            )

        except OSError:

            return JSONResponse(
                {
                    "message": (
                        "ComicCraft could not save the demo story. "
                        "Check available disk space and folder permissions."
                    )
                },
                status_code=500,
            )

        async def demo_events():

            yield _sse_event(
                {
                    "type": "progress",
                    "message": (
                        (
                            "Gemini API key is missing; "
                            "creating a clearly labeled demo "
                            "from your story."
                        )
                        if (
                            not GEMINI_API_KEY
                            and generation_mode != "demo"
                        )
                        else (
                            "Creating a clearly labeled "
                            "demo from your story."
                        )
                    ),
                    "progress": 30,
                }
            )

            yield _sse_event(
                {
                    "type": "complete",
                    "generation_id": (
                        generation.generation_id
                    ),
                    "is_demo": True,
                    "progress": 100,
                }
            )

        return StreamingResponse(
            demo_events(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-store",
                "X-Accel-Buffering": "no",
            },
        )

    # --------------------------------------------------------
    # Gemini generation lock
    # --------------------------------------------------------

    client_id = (
        request.client.host
        if request.client
        else "unknown"
    )

    limit_error = _reserve_generation_slot(
        client_id
    )

    if limit_error:

        return JSONResponse(
            {
                "message": limit_error[0]
            },
            status_code=limit_error[1],
        )

    generation_id = str(
        uuid.uuid4()
    )

    # --------------------------------------------------------
    # SSE generation events
    # --------------------------------------------------------

    async def events():

        try:

            # =================================================
            # STEP 1 — STORY
            # =================================================

            message = (
                "Writing your original story with Gemini..."
            )

            _console_progress(
                "▶ " + message
            )

            yield _sse_event(
                {
                    "type": "progress",
                    "message": message,
                    "progress": 5,
                }
            )

            story = await asyncio.to_thread(
                ai_service.generate_story,
                GEMINI_API_KEY,
                GEMINI_TEXT_MODEL,
                values,
            )

            _console_progress(
                "✓ Story generated successfully."
            )

            yield _sse_event(
                {
                    "type": "progress",
                    "message": (
                        "Story generated. "
                        "Preparing comic panels..."
                    ),
                    "progress": 20,
                }
            )

            # =================================================
            # STEP 2 — PANEL IMAGES
            # =================================================

            images: list[
                tuple[bytes, str]
            ] = []

            total_panels = len(
                story.scenes
            )

            _console_progress(
                f"Starting image generation: "
                f"{total_panels} panels."
            )

            for index in range(
                total_panels
            ):

                panel_number = index + 1

                message = (
                    f"🎨 Generating Panel "
                    f"{panel_number}/{total_panels}..."
                )

                _console_progress(
                    "▶ " + message
                )

                progress_start = (
                    25
                    + index
                    * (
                        68
                        / total_panels
                    )
                )

                yield _sse_event(
                    {
                        "type": "progress",
                        "message": message,
                        "progress": progress_start,
                    }
                )

                # ------------------------------------------------
                # Panel 1 has no reference.
                #
                # Panels 2+ use Panel 1 as the character/style
                # reference so the protagonist stays consistent.
                # ------------------------------------------------

                reference_bytes, reference_mime = (
                    images[0]
                    if images
                    else (
                        None,
                        "image/jpeg",
                    )
                )

                image = await asyncio.to_thread(
                    ai_service.generate_panel_image,
                    GEMINI_API_KEY,
                    GEMINI_IMAGE_MODEL,
                    story,
                    index,
                    reference_bytes,
                    reference_mime,
                )

                images.append(image)

                _console_progress(
                    f"✓ Panel "
                    f"{panel_number}/{total_panels} "
                    f"completed."
                )

                progress_complete = (
                    25
                    + panel_number
                    * (
                        68
                        / total_panels
                    )
                )

                yield _sse_event(
                    {
                        "type": "progress",
                        "message": (
                            f"✓ Panel "
                            f"{panel_number}/"
                            f"{total_panels} "
                            "is ready."
                        ),
                        "progress": progress_complete,
                    }
                )

            # =================================================
            # STEP 3 — SAVE
            # =================================================

            _console_progress(
                "Saving complete comic..."
            )

            generation = await asyncio.to_thread(
                _save_generation,
                generation_id,
                values,
                story,
                images,
            )

            _console_progress(
                "✓ Comic complete!"
            )

            _console_progress(
                f"Generation ID: "
                f"{generation.generation_id}"
            )

            yield _sse_event(
                {
                    "type": "complete",
                    "generation_id": (
                        generation.generation_id
                    ),
                    "progress": 100,
                }
            )

        # =====================================================
        # GEMINI ERROR
        # =====================================================

        except ai_service.GeminiServiceError as error:

            _console_progress(
                f"✗ Gemini error: {error}"
            )

            yield _sse_event(
                {
                    "type": "error",
                    "message": str(error),
                }
            )

        # =====================================================
        # OTHER ERROR
        # =====================================================

        except Exception as error:

            _console_progress(
                "✗ Unexpected error: "
                f"{type(error).__name__}: "
                f"{error}"
            )

            yield _sse_event(
                {
                    "type": "error",
                    "message": (
                        "ComicCraft could not save "
                        "the generated comic. "
                        "Check the VS Code terminal "
                        "for details."
                    ),
                }
            )

        # =====================================================
        # ALWAYS RELEASE GENERATION SLOT
        # =====================================================

        finally:

            _release_generation_slot()

            _console_progress(
                "Generation slot released."
            )

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================
# GENERATED PANEL IMAGE
# ============================================================

@app.get(
    "/generated/{generation_id}/panel/{panel_index}"
)
async def generated_panel(
    generation_id: str,
    panel_index: int,
) -> Response:

    generation = _load_generation(
        generation_id
    )

    if (
        generation is None
        or not 0 <= panel_index
        < len(generation.story.scenes)
    ):
        return JSONResponse(
            {
                "message": (
                    "This generated image is no longer available. "
                    "Start a new story to make another comic."
                )
            },
            status_code=404,
        )

    path = _image_path(
        generation,
        panel_index,
    )

    if path is None:
        return JSONResponse(
            {
                "message": (
                    "This generated image "
                    "is no longer available."
                )
            },
            status_code=404,
        )

    try:

        image_bytes = await asyncio.to_thread(
            path.read_bytes
        )

    except OSError:

        return JSONResponse(
            {
                "message": (
                    "This generated image "
                    "is no longer available."
                )
            },
            status_code=404,
        )

    return Response(
        image_bytes,
        media_type=(
            generation.image_mime_types[
                panel_index
            ]
        ),
        headers={
            "Cache-Control": (
                "private, max-age=3600"
            ),
            "X-Content-Type-Options": (
                "nosniff"
            ),
        },
    )


# ============================================================
# PANEL REGENERATION
# ============================================================

@app.post(
    "/api/generations/{generation_id}"
    "/panels/{panel_index}/regenerate"
)
async def regenerate_panel(
    request: Request,
    generation_id: str,
    panel_index: int,
) -> Response:

    generation = _load_generation(
        generation_id
    )

    if (
        generation is None
        or not 0 <= panel_index
        < len(generation.story.scenes)
    ):
        return JSONResponse(
            {
                "message": (
                    "This comic has expired. "
                    "Start a new story to generate more images."
                )
            },
            status_code=404,
        )

    if generation.is_demo:
        return JSONResponse(
            {
                "message": (
                    "Demo panels are text-only. "
                    "Use Gemini generation to create AI images."
                )
            },
            status_code=409,
        )

    if not GEMINI_API_KEY:
        return JSONResponse(
            {
                "message": (
                    "Image regeneration needs "
                    "GEMINI_API_KEY in the project-root .env file."
                )
            },
            status_code=503,
        )

    client_id = (
        request.client.host
        if request.client
        else "unknown"
    )

    limit_error = _reserve_generation_slot(
        client_id,
        image=True,
    )

    if limit_error:
        return JSONResponse(
            {
                "message": limit_error[0]
            },
            status_code=limit_error[1],
        )

    try:

        # Panel 2+ use panel 1 as reference.
        # If regenerating panel 1, use panel 2 instead.
        reference_index = (
            0
            if panel_index != 0
            else 1
        )

        reference_path = _image_path(
            generation,
            reference_index,
        )

        previous_panel_path = _image_path(
            generation,
            panel_index,
        )

        if (
            reference_path is None
            or previous_panel_path is None
        ):
            return JSONResponse(
                {
                    "message": (
                        "The character reference "
                        "image is unavailable."
                    )
                },
                status_code=404,
            )

        reference_bytes = (
            await asyncio.to_thread(
                reference_path.read_bytes
            )
        )

        image_bytes, mime_type = (
            await asyncio.to_thread(
                ai_service.generate_panel_image,
                GEMINI_API_KEY,
                GEMINI_IMAGE_MODEL,
                generation.story,
                panel_index,
                reference_bytes,
                generation.image_mime_types[
                    reference_index
                ],
            )
        )

        directory = _generation_directory(
            generation_id
        )

        if directory is None:
            return JSONResponse(
                {
                    "message": (
                        "This comic is no longer available."
                    )
                },
                status_code=404,
            )

        new_path = (
            directory
            / f"panel-{panel_index}"
            f"{_image_extension(mime_type)}"
        )

        temp_path = new_path.with_suffix(
            new_path.suffix + ".tmp"
        )

        await asyncio.to_thread(
            temp_path.write_bytes,
            image_bytes,
        )

        await asyncio.to_thread(
            temp_path.replace,
            new_path,
        )

        if new_path != previous_panel_path:
            previous_panel_path.unlink(
                missing_ok=True
            )

        generation.image_mime_types[
            panel_index
        ] = mime_type.lower()

        manifest_tmp = (
            directory / "manifest.tmp"
        )

        await asyncio.to_thread(
            manifest_tmp.write_text,
            generation.model_dump_json(),
            encoding="utf-8",
        )

        await asyncio.to_thread(
            manifest_tmp.replace,
            directory / "manifest.json",
        )

        return JSONResponse(
            {
                "image_url": (
                    f"/generated/"
                    f"{generation_id}"
                    f"/panel/{panel_index}"
                    f"?v={time.time_ns()}"
                )
            }
        )

    except ai_service.GeminiServiceError as error:

        return JSONResponse(
            {
                "message": str(error)
            },
            status_code=error.status_code,
        )

    except Exception:

        return JSONResponse(
            {
                "message": (
                    "Gemini could not regenerate "
                    "this panel. Please try again."
                )
            },
            status_code=502,
        )

    finally:

        _release_generation_slot()


# ============================================================
# NON-STREAMING STORY CREATION
# ============================================================

@app.post(
    "/preview",
    response_class=HTMLResponse,
)
@app.post(
    "/generate",
    response_class=HTMLResponse,
)
@app.post(
    "/regenerate",
    response_class=HTMLResponse,
)
async def create_story(
    request: Request,
    title: str = Form(""),
    character: str = Form(""),
    genre: str = Form("Fantasy"),
    description: str = Form(""),
    character_details: str = Form(""),
    setting: str = Form(""),
    tone: str = Form("Inspiring"),
    art_style: str = Form("Anime"),
    panel_count: str = Form("4"),
    generation_id: str = Form(""),
    generation_mode: str = Form("gemini"),
) -> HTMLResponse:

    values, error = _validate_story(
        title,
        character,
        genre,
        description,
        character_details,
        setting,
        tone,
        art_style,
        panel_count,
    )

    if error:

        return _render_page(
            request,
            values,
            error=error,
            status_code=422,
            stage="story",
        )

    # Existing generation requested.
    if generation_id:

        generation = _load_generation(
            generation_id
        )

        if (
            generation is None
            or generation.values != values
        ):
            return _render_page(
                request,
                values,
                error=(
                    "This generated comic has expired. "
                    "Generate it again before continuing."
                ),
                status_code=422,
                stage="story",
            )

        return _render_generation(
            request,
            generation,
            "preview",
        )

    # Demo mode.
    if (
        generation_mode == "demo"
        or not GEMINI_API_KEY
    ):

        try:

            generation = await asyncio.to_thread(
                _generate_demo_generation,
                values,
            )

        except OSError:

            return _render_page(
                request,
                values,
                error=(
                    "ComicCraft could not save the demo story. "
                    "Check available disk space and folder permissions."
                ),
                status_code=500,
                stage="story",
            )

        return _render_generation(
            request,
            generation,
            "preview",
        )

    # Gemini generation.
    client_id = (
        request.client.host
        if request.client
        else "unknown"
    )

    limit_error = _reserve_generation_slot(
        client_id
    )

    if limit_error:

        return _render_page(
            request,
            values,
            error=limit_error[0],
            status_code=limit_error[1],
            stage="story",
        )

    try:

        generation = await asyncio.to_thread(
            _generate_complete,
            values,
        )

    except ai_service.GeminiServiceError as generation_error:

        return _render_page(
            request,
            values,
            error=str(generation_error),
            status_code=generation_error.status_code,
            stage="story",
        )

    except Exception:

        return _render_page(
            request,
            values,
            error=(
                "ComicCraft could not save the generated images. "
                "Check available disk space, then try again."
            ),
            status_code=500,
            stage="story",
        )

    finally:

        _release_generation_slot()

    return _render_generation(
        request,
        generation,
        "preview",
    )


# ============================================================
# FINAL STORY
# ============================================================

@app.post(
    "/final",
    response_class=HTMLResponse,
)
async def final_story(
    request: Request,
    title: str = Form(""),
    character: str = Form(""),
    genre: str = Form("Fantasy"),
    description: str = Form(""),
    character_details: str = Form(""),
    setting: str = Form(""),
    tone: str = Form("Inspiring"),
    art_style: str = Form("Anime"),
    panel_count: str = Form("4"),
    generation_id: str = Form(""),
) -> HTMLResponse:

    values, error = _validate_story(
        title,
        character,
        genre,
        description,
        character_details,
        setting,
        tone,
        art_style,
        panel_count,
    )

    if error:

        return _render_page(
            request,
            values,
            error=error,
            status_code=422,
            stage="story",
        )

    generation = _load_generation(
        generation_id
    )

    if (
        generation is None
        or generation.values != values
    ):

        return _render_page(
            request,
            values,
            error=(
                "Generate a complete AI comic "
                "before opening the final story."
            ),
            status_code=422,
            stage="story",
        )

    return _render_generation(
        request,
        generation,
        "final",
    )


@app.get(
    "/final",
    response_class=HTMLResponse,
)
async def final_generated(
    request: Request,
    generation_id: str = "",
) -> Response:

    generation = _load_generation(
        generation_id
    )

    if generation is None:
        return RedirectResponse(
            "/story",
            status_code=303,
        )

    return _render_generation(
        request,
        generation,
        "final",
    )


# ============================================================
# PDF DOWNLOAD
# ============================================================

@app.post("/download")
async def download_comic(
    request: Request,
    title: str = Form(""),
    character: str = Form(""),
    genre: str = Form("Fantasy"),
    description: str = Form(""),
    character_details: str = Form(""),
    setting: str = Form(""),
    tone: str = Form("Inspiring"),
    art_style: str = Form("Anime"),
    panel_count: str = Form("4"),
    generation_id: str = Form(""),
) -> Response:

    values, error = _validate_story(
        title,
        character,
        genre,
        description,
        character_details,
        setting,
        tone,
        art_style,
        panel_count,
    )

    if error:

        return _render_page(
            request,
            values,
            error=error,
            status_code=422,
            stage="story",
        )

    generation = _load_generation(
        generation_id
    )

    if (
        generation is None
        or generation.values != values
    ):

        return _render_page(
            request,
            values,
            error=(
                "The generated comic images are "
                "no longer available. Generate "
                "the comic again before downloading."
            ),
            status_code=422,
            stage="story",
        )

    # ========================================================
    # PDF DOCUMENT
    # ========================================================

    pdf = BytesIO()

    page_width, page_height = letter

    document = SimpleDocTemplate(
        pdf,
        pagesize=letter,
        rightMargin=0.5 * inch,
        leftMargin=0.5 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
        title=generation.story.title,
        author="ComicCraft",
    )

    styles = getSampleStyleSheet()

    heading = ParagraphStyle(
        "ComicTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=25,
        leading=30,
        textColor=colors.HexColor(
            "#26143A"
        ),
        alignment=TA_LEFT,
        spaceAfter=4,
    )

    subtitle = ParagraphStyle(
        "ComicSubtitle",
        parent=styles["Normal"],
        fontSize=13,
        leading=17,
        textColor=colors.HexColor(
            "#59466C"
        ),
        spaceAfter=12,
    )

    story_style = ParagraphStyle(
        "ComicStory",
        parent=styles["Normal"],
        fontSize=12,
        leading=17,
        textColor=colors.HexColor(
            "#261F2D"
        ),
        spaceAfter=15,
    )

    beat_style = ParagraphStyle(
        "Beat",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=13,
        textColor=colors.HexColor(
            "#D0A5FF"
        ),
        spaceAfter=5,
    )

    demo_style = ParagraphStyle(
        "DemoPanel",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor(
            "#7043A0"
        ),
        spaceAfter=8,
    )

    caption_style = ParagraphStyle(
        "Caption",
        parent=styles["Normal"],
        fontSize=11,
        leading=15,
        textColor=colors.white,
        spaceAfter=7,
    )

    dialogue_style = ParagraphStyle(
        "Dialogue",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor(
            "#C9B8E8"
        ),
    )

    # ========================================================
    # PDF HEADER / STORY
    # ========================================================

    story = [
        Paragraph(
            escape(
                generation.story.title
            ),
            heading,
        ),

        Paragraph(
            (
                f"{escape(values['genre'])} "
                f"comic starring "
                f"{escape(values['character'])}"
            ),
            subtitle,
        ),

        Paragraph(
            "<b>Character</b>: "
            + escape(
                values["character"]
            )
            + "<br/>"
            + "<b>Character details</b>: "
            + escape(
                values.get(
                    "character_details"
                )
                or generation.story.character_description
            )
            + "<br/>"
            + "<b>Generated visual guide</b>: "
            + escape(
                generation.story.character_description
            )
            + "<br/>"
            + "<b>Setting</b>: "
            + escape(
                values.get("setting")
                or "Created from the story prompt"
            ),
            story_style,
        ),

        Paragraph(
            escape(
                generation.story.story_text
            ).replace(
                "\n",
                "<br/>",
            ),
            story_style,
        ),
    ]

    # ========================================================
    # PDF PANELS
    # ========================================================

    panel_cells = []

    cell_width = (
        page_width
        - document.leftMargin
        - document.rightMargin
    ) / 2

    for index, panel in enumerate(
        generation.story.scenes
    ):

        cell = []

        if generation.is_demo:

            cell.append(
                Paragraph(
                    (
                        "DEMO STORYBOARD PANEL — "
                        "no AI image generated"
                    ),
                    demo_style,
                )
            )

        else:

            image_path = _image_path(
                generation,
                index,
            )

            if image_path is None:

                return _render_page(
                    request,
                    values,
                    error=(
                        "A generated panel image "
                        "is unavailable. "
                        "Generate the comic again "
                        "before downloading."
                    ),
                    status_code=422,
                    stage="story",
                )

            image_stream = BytesIO(
                await asyncio.to_thread(
                    image_path.read_bytes
                )
            )

            cell.extend(
                [
                    PdfImage(
                        image_stream,
                        width=(
                            cell_width
                            - 0.42 * inch
                        ),
                        height=2.25 * inch,
                        kind="proportional",
                    ),
                    Spacer(1, 8),
                ]
            )

        cell.extend(
            [
                Paragraph(
                    escape(
                        PANEL_BEATS[
                            index
                            % len(PANEL_BEATS)
                        ]
                    ),
                    beat_style,
                ),

                Paragraph(
                    escape(
                        panel.caption
                    ),
                    caption_style,
                ),

                Paragraph(
                    escape(
                        panel.dialogue
                    ),
                    dialogue_style,
                ),
            ]
        )

        panel_cells.append(cell)

    # ========================================================
    # PANEL GRID
    # ========================================================

    panel_rows = [
        panel_cells[index:index + 2]
        for index in range(
            0,
            len(panel_cells),
            2,
        )
    ]

    if (
        panel_rows
        and len(panel_rows[-1]) == 1
    ):
        panel_rows[-1].append("")

    comic_grid = Table(
        panel_rows,
        colWidths=[
            cell_width,
            cell_width,
        ],
        hAlign="LEFT",
    )

    comic_grid.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    colors.HexColor(
                        "#1A1028"
                    ),
                ),

                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    1.2,
                    colors.HexColor(
                        "#7043A0"
                    ),
                ),

                (
                    "INNERGRID",
                    (0, 0),
                    (-1, -1),
                    1.2,
                    colors.HexColor(
                        "#7043A0"
                    ),
                ),

                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
                ),

                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    16,
                ),

                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    16,
                ),

                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    12,
                ),

                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    12,
                ),
            ]
        )
    )

    story.append(
        comic_grid
    )

    # ========================================================
    # BUILD PDF
    # ========================================================

    document.build(story)

    pdf.seek(0)

    filename = (
        re.sub(
            r"[^A-Za-z0-9_-]+",
            "-",
            values["title"],
        )
        .strip("-")
        .lower()
        or "comiccraft-story"
    )

    return StreamingResponse(
        pdf,
        media_type="application/pdf",
        headers={
            "Content-Disposition": (
                f'attachment; '
                f'filename="{filename}.pdf"'
            )
        },
    )