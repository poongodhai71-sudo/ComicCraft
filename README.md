**ComicCraft – AI Comic Story Creator**

ComicCraft is an AI-powered web application that creates original comic stories and illustrated comic panels from a user's idea.

The application allows users to enter a story prompt, character details, genre, tone, art style, and number of panels. Google Gemini generates the comic story, while Stability AI generates the illustrations for the individual comic panels.

The completed comic can be previewed and exported as a PDF.

**Features**
AI-generated comic stories
Custom story prompts
Character and setting details
Genre and tone selection
Multiple art styles
2–6 comic panels
AI-generated panel illustrations
Character information included across panels
Comic preview
Story regeneration
Individual panel image regeneration
PDF export
Secure server-side API key handling

Technologies Used
Python
FastAPI
Google Gemini API
Stability AI
HTML
CSS
JavaScript
Pydantic
Uvicorn


**How It Works:**

User Input
    ↓
FastAPI Backend
    ↓
Google Gemini
    ↓
Structured Comic Story
    ↓
Panel / Scene Prompts
    ↓
Stability AI
    ↓
Generated Comic Images
    ↓
Comic Preview
    ↓
PDF Export

**Project Structure**

ComicCraft/
│
├── static/
│   ├── app.js
│   └── style.css
│
├── templates/
│   └── index.html
│
├── tests/
│   ├── test_ai_workflow.py
│   └── test_main.py
│
├── .env.example
├── .gitignore
├── ai_service.py
├── main.py
├── Procfile
├── requirements.txt
└── README.md

**AI Models**

Story Generation

ComicCraft uses:
gemini-3.5-flash-lite

Google Gemini is used to generate the structured comic story, including scenes, captions, dialogue, and character information.

**Image Generation**

ComicCraft uses Stability AI to generate the illustrations for the comic panels.

Each panel is generated using the scene and character information produced during the story-generation process.

**Requirements**

Windows 10 or Windows 11
Python 3.10 or newer
Google Gemini API key
Stability AI API key
Internet connection

**Installation**

1. Clone the repository
git clone https://github.com/poongodhai71-sudo/ComicCraft.git
cd ComicCraft

2. Create a virtual environment
py -m venv .venv
3. Activate the virtual environment
.\.venv\Scripts\Activate.ps1
4. Install dependencies
python -m pip install -r requirements.txt

**Environment Variables**
Create a .env file in the project folder:

GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_TEXT_MODEL=gemini-3.5-flash-lite
STABILITY_API_KEY=your_stability_api_key_here

**Run the Application**
From the C:\ComicCraft folder, run:

.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8768
Then open:
http://127.0.0.1:8768/story

**Creating a Comic**

1.Enter a story title.
2.Enter your story idea.
3.Enter character details.
4.Select a genre and tone.
5.Select an art style.
6.Choose 2–6 panels.
7.Click Generate My Comic.
8.Wait for the story and images to be generated.
9.Preview the completed comic.
10.Download the comic as a PDF.

**PDF Export**

ComicCraft can export the generated comic as a PDF containing the story content, panel illustrations, captions, and dialogue.

**Security**

API keys are stored in environment variables and are not included in the application's frontend code.

The .env file is excluded from Git using .gitignore.

**Future Improvements**

Improved character consistency
More comic layouts
More art styles
User accounts
Cloud storage for generated comics
Improved mobile support
Additional comic customization options


**Author**

**Poongodhai**
ComicCraft – AI Comic Story Creator
