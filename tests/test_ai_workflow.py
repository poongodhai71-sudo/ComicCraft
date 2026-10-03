from __future__ import annotations

import json
import re
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
from google.genai.errors import APIError
from fastapi.testclient import TestClient
from PIL import Image

import ai_service
import main


FORM = {
    "title": "Moonlit Archive",
    "character": "Sena",
    "genre": "Mystery",
    "description": "Sena finds a silver bookmark in a locked archive. It points to a shelf that should not exist. The final page contains a map of tomorrow.",
}
STORY = ai_service.ComicStory(
    title="Moonlit Archive",
    character_description="Sena has a silver bob, a plum coat, and a crescent hair clip.",
    story_text="Sena finds a silver bookmark in a locked archive. It leads her to a hidden shelf and a map of tomorrow.",
    scenes=[
        ai_service.ComicScene(
            scene_description=f"Sena investigates scene {index + 1} in the moonlit archive.",
            caption=f"Scene {index + 1} begins.",
            dialogue=f"Sena: clue {index + 1}.",
            image_prompt=f"Sena discovers clue {index + 1}, wearing her plum coat.",
        )
        for index in range(4)
    ],
)


def png_bytes(color: tuple[int, int, int]) -> bytes:
    image = Image.new("RGB", (96, 72), color)
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


class AiWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.patchers = [
            patch.object(main, "GENERATION_DIR", Path(self.temp_dir.name)),
            patch.object(main, "_generation_cache", {}),
            patch.object(main, "GEMINI_API_KEY", "test-key-not-real"),
            patch.object(main, "GENERATION_COOLDOWN_SECONDS", 0),
            patch.object(main, "IMAGE_REGENERATION_COOLDOWN_SECONDS", 0),
            patch.object(main, "MAX_ACTIVE_GENERATIONS", 2),
        ]
        for patcher in self.patchers:
            patcher.start()
        main._active_generations = 0
        main._last_generation_by_client.clear()
        main._last_image_generation_by_client.clear()
        self.client = TestClient(main.app)

    def tearDown(self):
        self.client.close()
        for patcher in reversed(self.patchers):
            patcher.stop()
        self.temp_dir.cleanup()

    def _generate(self, image_side_effect=None, form=None, story=None):
        images = [png_bytes((index * 40, 35, 140)) for index in range(4)]

        def default_image(_key, _model, _story, scene_index, _reference=None, _mime="image/jpeg"):
            return images[scene_index], "image/png"

        with patch("main.ai_service.generate_story", return_value=story or STORY) as story_call:
            with patch("main.ai_service.generate_panel_image", side_effect=image_side_effect or default_image) as image_call:
                response = self.client.post("/api/generate", data=form or FORM)
        self.assertEqual(response.status_code, 200)
        events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
        return response, events, story_call, image_call, images

    def test_four_real_provider_calls_render_and_export_generated_images(self):
        _response, events, story_call, image_call, images = self._generate()
        complete = next(event for event in events if event["type"] == "complete")
        generation_id = complete["generation_id"]
        self.assertEqual(story_call.call_count, 1)
        self.assertEqual(image_call.call_count, 4)
        self.assertEqual([call.args[3] for call in image_call.call_args_list], [0, 1, 2, 3])
        self.assertIsNone(image_call.call_args_list[0].args[4])
        for call in image_call.call_args_list[1:]:
            self.assertEqual(call.args[4], images[0])
        self.assertEqual(sum(event.get("type") == "progress" for event in events), 10)

        preview = self.client.get(f"/preview?generation_id={generation_id}")
        self.assertEqual(preview.status_code, 200)
        self.assertIn("Sena finds a silver bookmark", preview.text)
        self.assertEqual(preview.text.count('class="panel-image"'), 4)
        self.assertNotIn("test-key-not-real", preview.text)
        for index, image in enumerate(images):
            served = self.client.get(f"/generated/{generation_id}/panel/{index}")
            self.assertEqual(served.status_code, 200)
            self.assertEqual(served.headers["content-type"], "image/png")
            self.assertEqual(served.content, image)

        final = self.client.post("/final", data={**FORM, "generation_id": generation_id})
        self.assertEqual(final.status_code, 200)
        self.assertIn("STEP 04", final.text)
        self.assertIn("Create New Story", final.text)
        self.assertIn("Download PDF", final.text)
        self.assertNotIn("Regenerate Story", final.text)

        pdf = self.client.post("/download", data={**FORM, "generation_id": generation_id})
        self.assertEqual(pdf.status_code, 200)
        self.assertEqual(pdf.headers["content-type"], "application/pdf")
        self.assertTrue(pdf.content.startswith(b"%PDF-"))
        self.assertIn(b"Moonlit Archive", pdf.content)
        self.assertEqual(pdf.content.count(b"/Subtype /Image"), 4)

    def test_selected_panel_count_and_story_details_survive_preview_final_and_pdf(self):
        selected_form = {
            **FORM,
            "character_details": "Silver hair, a plum coat, and a curious nature.",
            "setting": "A hidden archive beneath the city.",
            "tone": "Emotional",
            "art_style": "Manga",
            "panel_count": "2",
        }
        two_scene_story = STORY.model_copy(update={"scenes": STORY.scenes[:2]})
        _, events, story_call, image_call, _ = self._generate(form=selected_form, story=two_scene_story)
        generation_id = next(event["generation_id"] for event in events if event["type"] == "complete")
        self.assertEqual(image_call.call_count, 2)
        self.assertEqual(story_call.call_args.args[2]["panel_count"], "2")
        self.assertEqual(story_call.call_args.args[2]["art_style"], "Manga")

        preview = self.client.get(f"/preview?generation_id={generation_id}")
        self.assertEqual(preview.status_code, 200)
        self.assertEqual(preview.text.count('class="panel-image"'), 2)
        self.assertIn(selected_form["character_details"], preview.text)
        self.assertIn(selected_form["setting"], preview.text)

        edit = self.client.post("/edit", data=selected_form)
        self.assertEqual(edit.status_code, 200)
        self.assertIn('value="2"', edit.text)
        final = self.client.post("/final", data={**selected_form, "generation_id": generation_id})
        self.assertEqual(final.status_code, 200)
        self.assertIn(selected_form["character_details"], final.text)

        pdf = self.client.post("/download", data={**selected_form, "generation_id": generation_id})
        self.assertEqual(pdf.status_code, 200)
        self.assertTrue(pdf.content.startswith(b"%PDF-"))
        self.assertEqual(pdf.content.count(b"/Subtype /Image"), 2)

    def test_welcome_page_has_requested_team_roster_and_entry_navigation(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Turn Your Imagination", response.text)
        self.assertIn("Start Creating", response.text)
        self.assertIn("Poongodhai B", response.text)
        for member in ("Pavatharani K", "Manikandan S", "Vasanthakumar K", "Subashree K"):
            self.assertIn(member, response.text)
        self.assertNotIn("Nitish Kumar T", response.text)

    def test_panel_regeneration_replaces_only_after_a_successful_image(self):
        _, events, _, _, images = self._generate()
        generation_id = next(event["generation_id"] for event in events if event["type"] == "complete")
        replacement = png_bytes((220, 20, 90))
        with patch("main.ai_service.generate_panel_image", return_value=(replacement, "image/png")) as image_call:
            response = self.client.post(f"/api/generations/{generation_id}/panels/2/regenerate")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(image_call.call_args.args[3], 2)
        self.assertEqual(image_call.call_args.args[4], images[0])
        self.assertEqual(self.client.get(f"/generated/{generation_id}/panel/2").content, replacement)
        self.assertEqual(self.client.get(f"/generated/{generation_id}/panel/1").content, images[1])

    def test_failed_panel_generation_saves_no_partial_comic(self):
        def fail_on_third(_key, _model, _story, scene_index, *_args):
            if scene_index == 2:
                raise ai_service.GeminiServiceError("Gemini rate limit reached.", 429)
            return png_bytes((40, scene_index * 30, 160)), "image/png"

        _, events, _, _, _ = self._generate(fail_on_third)
        self.assertFalse(any(event["type"] == "complete" for event in events))
        self.assertEqual(events[-1]["type"], "error")
        self.assertIn("rate limit", events[-1]["message"].lower())
        self.assertEqual(list(Path(self.temp_dir.name).iterdir()), [])

    def test_missing_key_is_explicit_and_uses_labeled_demo_through_pdf(self):
        with patch.object(main, "GEMINI_API_KEY", ""):
            with patch("main.ai_service.generate_story", side_effect=AssertionError("demo must not call Gemini")) as story_call:
                with patch("main.ai_service.generate_panel_image", side_effect=AssertionError("demo must not call Gemini")) as image_call:
                    response = self.client.post("/api/generate", data=FORM)
        self.assertEqual(response.status_code, 200)
        story_call.assert_not_called()
        image_call.assert_not_called()
        events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
        self.assertIn("API key is missing", events[0]["message"])
        self.assertEqual(events[-1]["type"], "complete")
        self.assertTrue(events[-1]["is_demo"])
        generation_id = events[-1]["generation_id"]
        self.assertEqual(list(Path(self.temp_dir.name).iterdir()), [Path(self.temp_dir.name) / generation_id])

        preview = self.client.get(f"/preview?generation_id={generation_id}")
        self.assertEqual(preview.status_code, 200)
        self.assertIn("DEMO MODE", preview.text)
        self.assertIn("No AI image generated", preview.text)
        self.assertIn("Configure a valid Gemini API key", preview.text)
        self.assertNotIn('class="panel-image"', preview.text)

        final = self.client.post("/final", data={**FORM, "generation_id": generation_id})
        self.assertEqual(final.status_code, 200)
        self.assertIn("STEP 04", final.text)
        pdf = self.client.post("/download", data={**FORM, "generation_id": generation_id})
        self.assertEqual(pdf.status_code, 200)
        self.assertTrue(pdf.content.startswith(b"%PDF-"))
        self.assertIn(b"Moonlit Archive", pdf.content)
        self.assertEqual(pdf.content.count(b"/Subtype /Image"), 0)
        self.assertEqual(self.client.get("/.env").status_code, 404)

    def test_explicit_demo_mode_works_even_when_gemini_is_configured(self):
        with patch("main.ai_service.generate_story", side_effect=AssertionError("demo must not call Gemini")) as story_call:
            with patch("main.ai_service.generate_panel_image", side_effect=AssertionError("demo must not call Gemini")) as image_call:
                response = self.client.post("/api/generate", data={**FORM, "generation_mode": "demo", "panel_count": "2"})
        self.assertEqual(response.status_code, 200)
        story_call.assert_not_called()
        image_call.assert_not_called()
        events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
        generation_id = events[-1]["generation_id"]
        self.assertTrue(events[-1]["is_demo"])

        preview = self.client.get(f"/preview?generation_id={generation_id}")
        self.assertEqual(preview.status_code, 200)
        self.assertEqual(preview.text.count('class="comic-panel"'), 2)
        self.assertNotIn('class="panel-image"', preview.text)
        self.assertIn("text-only demo", preview.text)
        edit = self.client.post("/edit", data=FORM)
        self.assertEqual(edit.status_code, 200)
        self.assertIn(FORM["title"], edit.text)
        final = self.client.post("/final", data={**FORM, "generation_id": generation_id, "panel_count": "2"})
        self.assertEqual(final.status_code, 200)
        self.assertIn(FORM["title"], final.text)
        self.assertIn("STEP 04", final.text)
        self.assertEqual(
            self.client.post(f"/api/generations/{generation_id}/panels/0/regenerate").status_code,
            409,
        )

    def test_sample_story_button_is_available_on_form(self):
        response = self.client.get("/story")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Fill a sample story", response.text)
        self.assertIn("Try Demo Mode", response.text)
        self.assertIn('<form action="/generate" method="post"', response.text)

    def test_native_create_story_route_runs_demo_and_final_story(self):
        with patch.object(main, "GEMINI_API_KEY", ""):
            with patch("main.ai_service.generate_story", side_effect=AssertionError("demo must not call Gemini")) as story_call:
                with patch("main.ai_service.generate_panel_image", side_effect=AssertionError("demo must not call Gemini")) as image_call:
                    response = self.client.post("/generate", data={**FORM, "generation_mode": "demo"})
        self.assertEqual(response.status_code, 200)
        story_call.assert_not_called()
        image_call.assert_not_called()
        self.assertIn("STEP 03", response.text)
        self.assertIn("DEMO MODE", response.text)
        generation_id = re.search(r'data-generation-id="([0-9a-f-]{36})"', response.text).group(1)
        final = self.client.post("/final", data={**FORM, "generation_id": generation_id})
        self.assertEqual(final.status_code, 200)
        self.assertIn("STEP 04", final.text)

    def test_environment_configuration_points_to_project_root_dotenv(self):
        self.assertEqual(main.ENV_FILE, main.BASE_DIR / ".env")
        self.assertTrue(main.ENV_FILE.is_file())
        self.assertEqual(main.GEMINI_TEXT_MODEL, ai_service.TEXT_MODEL_DEFAULT)
        self.assertEqual(ai_service.TEXT_MODEL_DEFAULT, "gemini-2.5-flash")
        self.assertEqual(ai_service.IMAGE_MODEL_DEFAULT, "gemini-3.1-flash-image")

    def test_official_sdk_configuration_uses_json_story_schema_and_image_output(self):
        image = png_bytes((90, 30, 170))
        story_values = {
            **FORM,
            "character_details": "Silver hair, a plum coat.",
            "setting": "A hidden archive.",
            "tone": "Emotional",
            "art_style": "Manga",
            "panel_count": "4",
        }
        text_client = MagicMock()
        image_client = MagicMock()
        text_client.__enter__.return_value = text_client
        image_client.__enter__.return_value = image_client
        text_client.models.generate_content.return_value = MagicMock(parsed=STORY, text=STORY.model_dump_json())
        image_part = MagicMock(inline_data=MagicMock(data=image, mime_type="image/png"))
        image_client.models.generate_content.return_value = MagicMock(parts=[image_part])

        with patch("ai_service.genai.Client", side_effect=[text_client, image_client]):
            story = ai_service.generate_story("test-key", ai_service.TEXT_MODEL_DEFAULT, story_values)
            result_image, mime_type = ai_service.generate_panel_image(
                "test-key", "gemini-3.1-flash-image", story, 0,
                reference_image=image, reference_mime_type="image/png",
            )

        self.assertEqual(story.title, story_values["title"])
        text_config = text_client.models.generate_content.call_args.kwargs["config"]
        prompt = text_client.models.generate_content.call_args.kwargs["contents"]
        self.assertIn(story_values["character_details"], prompt)
        self.assertIn(story_values["setting"], prompt)
        self.assertIn(story_values["tone"], prompt)
        self.assertIn(story_values["art_style"], prompt)
        self.assertIn("exactly 4 ordered scenes", prompt)
        self.assertEqual(text_config.response_mime_type, "application/json")
        self.assertIs(text_config.response_schema, ai_service.ComicStory)
        image_config = image_client.models.generate_content.call_args.kwargs["config"]
        self.assertEqual(image_config.response_modalities, ["IMAGE"])
        self.assertEqual(image_config.image_config.aspect_ratio, "4:3")
        self.assertEqual(result_image, image)
        self.assertEqual(mime_type, "image/png")
        self.assertEqual(ai_service.TEXT_MODEL_DEFAULT, "gemini-2.5-flash")
        self.assertEqual(ai_service.IMAGE_MODEL_DEFAULT, "gemini-3.1-flash-image")

    def test_invalid_gemini_key_error_is_actionable_and_keeps_demo_available(self):
        provider_error = APIError(400, {
            "error": {
                "code": 400,
                "message": "API key not valid. Please pass a valid API key.",
                "status": "INVALID_ARGUMENT",
            },
        })
        friendly = ai_service._friendly_api_error(provider_error)
        self.assertEqual(friendly.status_code, 503)
        self.assertIn("GEMINI_API_KEY", str(friendly))
        self.assertIn("Try Demo Mode", str(friendly))

        with patch("main.ai_service.generate_story", side_effect=friendly):
            response = self.client.post("/api/generate", data=FORM)
        events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
        self.assertEqual(events[-1]["type"], "error")
        self.assertIn("project-root .env", events[-1]["message"])
        form = self.client.get("/story")
        self.assertIn("Try Demo Mode", form.text)

    def test_gemini_quota_model_and_network_errors_are_actionable(self):
        quota_error = APIError(429, {
            "error": {
                "code": 429,
                "message": "Quota exceeded.",
                "status": "RESOURCE_EXHAUSTED",
            },
        })
        quota = ai_service._friendly_api_error(quota_error)
        self.assertEqual(quota.status_code, 429)
        self.assertIn("quota", str(quota).lower())

        model_error = APIError(404, {
            "error": {
                "code": 404,
                "message": "Model not found.",
                "status": "NOT_FOUND",
            },
        })
        model = ai_service._friendly_api_error(model_error)
        self.assertEqual(model.status_code, 502)
        self.assertIn("model", str(model).lower())

        request = httpx.Request("POST", "https://generativelanguage.googleapis.com/")
        network = ai_service._friendly_api_error(httpx.ConnectError("network unavailable", request=request))
        self.assertEqual(network.status_code, 503)
        self.assertIn("internet connection", str(network).lower())

    def test_legacy_generate_route_runs_ai_and_invalid_input_stays_validated(self):
        images = [png_bytes((20, 100, index * 30)) for index in range(4)]

        def fake_image(_key, _model, _story, scene_index, reference_image=None, reference_mime_type="image/jpeg"):
            return images[scene_index], "image/png"

        with patch("main.ai_service.generate_story", return_value=STORY):
            with patch("main.ai_service.generate_panel_image", side_effect=fake_image):
                response = self.client.post("/generate", data=FORM)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.text.count('class="panel-image"'), 4)

        invalid = self.client.post("/api/generate", data={**FORM, "title": ""})
        self.assertEqual(invalid.status_code, 422)
        self.assertIn("title", invalid.json()["message"].lower())


if __name__ == "__main__":
    unittest.main()
