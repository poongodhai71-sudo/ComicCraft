import unittest

from main import DEMO_STORY, GENRE_CHOICES, _validate_story, generate_comic


class ComicGenerationTests(unittest.TestCase):
    def test_demo_generates_four_personalized_panels(self):
        panels = generate_comic(**DEMO_STORY)

        self.assertEqual(len(panels), 4)
        self.assertIn(DEMO_STORY["character"], panels[0]["caption"])
        self.assertIn("lantern", panels[0]["caption"])
        self.assertTrue(all(panel["caption"] and panel["dialogue"] for panel in panels))

    def test_story_validation_rejects_missing_and_oversized_fields(self):
        values, error = _validate_story("", "Mira", "Fantasy", "A story")
        self.assertIsNotNone(error)
        self.assertEqual(values["character"], "Mira")

        _, error = _validate_story("A" * 81, "Mira", "Fantasy", "A story")
        self.assertIsNotNone(error)

    def test_story_validation_rejects_unknown_genre(self):
        _, error = _validate_story("A title", "Mira", "Western", "A story")
        self.assertIsNotNone(error)
        self.assertIn("Fantasy", GENRE_CHOICES)

    def test_story_validation_preserves_style_details_and_rejects_unknown_panel_count(self):
        values, error = _validate_story(
            "A title", "Mira", "Fantasy", "A story", "Silver armor", "Moonlit bay",
            "Funny", "Watercolor", "6",
        )
        self.assertIsNone(error)
        self.assertEqual(values["character_details"], "Silver armor")
        self.assertEqual(values["setting"], "Moonlit bay")
        self.assertEqual(values["panel_count"], "6")

        _, error = _validate_story("A title", "Mira", "Fantasy", "A story", panel_count="8")
        self.assertIsNotNone(error)

    def test_extra_description_sentences_still_make_four_panels(self):
        panels = generate_comic("A title", "Mira", "Mystery", "First. Second. Third. Fourth.")
        self.assertEqual(len(panels), 4)
        self.assertIn("Fourth", panels[3]["caption"])


if __name__ == "__main__":
    unittest.main()
