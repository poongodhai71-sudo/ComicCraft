# ComicCraft - AI Comic Story Creator

A FastAPI comic creator using Google's official Gemini API for an original story and a user-selected set of 2–6 illustrated comic panels, with a PDF export. Story details and panel images are stored server-side; the API key is never sent to browser code or responses.

## Requirements

- Windows 10 or 11
- Python 3.10 or newer
- A valid Gemini API key with access to the configured text and image models

## Run on Windows

Create a key in [Google AI Studio](https://aistudio.google.com/apikey), then open PowerShell in this folder and run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
notepad .env
python -m uvicorn main:app --reload
```

In `.env`, replace the empty `GEMINI_API_KEY=` value with an active key created in [Google AI Studio](https://aistudio.google.com/apikey). Keep the key on that line, with no placeholder text. If Gemini reports `API key not valid`, create a fresh key for a project with the Gemini API enabled, replace the value in `.env`, save, and restart Uvicorn. A key that exists in `.env` can still be revoked, malformed, restricted, or otherwise invalid; ComicCraft cannot repair or create the credential. Do not paste it into chat, source code, browser fields, or a public repository. The app reads `.env` from the project directory containing `main.py`; an explicitly-set process environment variable takes precedence.

Open <http://127.0.0.1:8000> in your browser. Leave the PowerShell window open while using the app. Stop the server with `Ctrl+C`.

If PowerShell blocks virtual-environment activation, use Command Prompt instead:

```bat
py -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if not exist .env copy .env.example .env
notepad .env
python -m uvicorn main:app --reload
```

## Production deployment and custom domain

ComicCraft is a FastAPI ASGI application. The root-level `Procfile` defines the production web process:

```sh
uvicorn main:app --host 0.0.0.0 --port $PORT
```

Deploy the existing project directory to a Python web-app host that supports FastAPI (for example, Render, Railway, or Heroku). Use `pip install -r requirements.txt` as its build/install command and the `Procfile` web process as its start command. The host must provide a `PORT` environment variable and route public HTTPS traffic to the service. Set `GEMINI_API_KEY` as a protected environment variable in the host dashboard; do not upload `.env` or commit secrets.

To map `https://comiccraft.one8.com/`, first deploy the app and add `comiccraft.one8.com` as a custom domain on that hosting service. The service will show its required DNS record (usually a CNAME target, or an A/AAAA record). In the Google Cloud DNS zone for `one8.com`, create the exact record the hosting service specifies for the `comiccraft` subdomain. Do not guess the record target or point it at an unrelated IP. Wait for DNS propagation and the host to provision TLS, then verify the HTTPS URL and the home, `/story`, `/preview`, and `/final` routes.

Public DNS currently reports `NXDOMAIN` for `comiccraft.one8.com`, so there is no DNS mapping to a deployed service yet. The application repository cannot create the hosting service, register the custom domain with a hosting account, or edit the Google Cloud DNS zone. Configure these in the relevant provider dashboards. Since generated comics are stored under `.comiccraft/generations`, configure persistent storage for that directory if generated files must survive service restarts or redeployments.

## Use

Enter the story title, prompt, character, and optional character/setting details. Choose a genre, tone, art style, and 2–6 panels, then select **Generate My Comic**. Gemini generates the story and requested panel images sequentially. **Regenerate Story** requests a new story and comic; **Regenerate Image** replaces a single panel; **Download PDF** exports the current story, character details, captions, dialogue, and generated images.

## Models and cost

- Text default: `gemini-2.5-flash` (`GEMINI_TEXT_MODEL` can override it).
- Image default: `gemini-3.1-flash-image` (Nano Banana 2, 1K images; `GEMINI_IMAGE_MODEL` can override it).
- The project uses Google's official `google-genai` Python SDK and Gemini structured output.
- Google's current paid-tier price for a 1K Gemini 3.1 Flash Image is about **$0.067 per image**, about **$0.268 for four**, plus text input/output usage. The image model has no free-tier API price listed on the current pricing page, so enabled billing may be required. Check [Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing) and your [AI Studio rate limits](https://aistudio.google.com/rate-limit); prices and quotas can change.
- A comic makes one text request and one image request per selected panel. If any request fails, ComicCraft does not publish or export a partial comic. Character consistency is guided by a repeated description and the first panel as a reference, but exact visual identity is not guaranteed.
- Generated files are stored locally under `.comiccraft/generations`, expire after 12 hours, and are pruned to at most eight saved comics. The app permits one active generation per server process and applies short per-client cooldowns.

When `GEMINI_API_KEY` is valid, **Generate My Comic** uses Gemini for the story and requested panel illustrations. The key is loaded from the project-root `.env` file on startup; after changing `.env`, restart Uvicorn. Invalid credentials, unavailable models, quota limits, and provider/network failures show actionable errors. Never commit or expose the key.

If the key is missing, the primary generation action creates a clearly labeled, input-based demo story so the preview, final-story, and PDF workflow can still be tried. Select **Try Demo Mode** to use the same local-only flow even when a key is configured; demo mode does not call Gemini. Demo storyboard panels are text-only and are explicitly not presented as AI-generated artwork. **Fill a sample story** populates the form with example input.

For the local server command used by this checkout, open PowerShell in `C:\ComicCraft` and run:

```powershell
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8768
```

Open <http://127.0.0.1:8768/story> to create a comic. Stop the server with `Ctrl+C` before restarting it after `.env` changes.

## Run tests

```powershell
python -m unittest discover -s tests -v
```
