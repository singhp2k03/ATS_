<p align="center">
  <h1 align="center">⚡ATS Engine</h1>
  <p align="center">
    <strong>AI-Powered Batch Resume Screening & Candidate Ranking System</strong>
  </p>
  <p align="center">
    <a href="#features">Features</a> •
    <a href="#architecture">Architecture</a> •
    <a href="#getting-started">Getting Started</a> •
    <a href="#api-reference">API Reference</a> •
    <a href="#configuration">Configuration</a>
  </p>
</p>

---

## 🎯 Overview

ATS Engine is an intelligent Applicant Tracking System that leverages **Google Gemini AI** to automate the resume screening process. Upload a batch of resumes, paste a job description, and get instant AI-powered candidate evaluations ranked by fit — complete with location-based commute analysis, skill matching, and detailed scoring breakdowns.

Built for recruiters and HR teams who need to screen hundreds of resumes quickly without sacrificing evaluation quality.

---

## ✨ Features

### 🤖 AI-Powered Evaluation
- **Structured scoring** across three pillars: Experience (40pts), Skills (30pts), Education (30pts)
- **Gemini 3.1 Flash Lite** for fast, cost-effective evaluations with structured JSON output
- **Auto JD extraction** — paste a job description and AI automatically populates experience range, skills, education, and location fields

### 📍 Hybrid Location Engine
- **Zero-cost geolocation** using OpenStreetMap's Nominatim geocoder
- **Real driving time** estimates via OSRM (Open Source Routing Machine)
- **Haversine distance** fallback when routing APIs hit limits
- **Smart relevancy tiers**: High / Medium / Low / Relocation based on commute time

### 🔁 Intelligent Deduplication
- **Tier 1 — Byte-Level**: MD5 hashing catches identical file uploads
- **Tier 2 — Identity Matching**: Merges duplicates by email/name, keeping the highest-scoring version

### 🎛️ Configurable Filters
- **Experience range** (min & max years)
- **Required skills**, **education**, and **target location**
- **Mandatory (hard) filters** — toggle any criterion as a hard pass/fail gate
- **Passing score threshold** — candidates below the threshold are auto-rejected

### 🖥️ Modern Web Interface
- Drag & drop resume upload (PDF/DOCX)
- Dark / Light theme toggle with persistence
- Skeleton loading states during processing
- Detailed candidate modal with pillar breakdown
- JD Library — save and reload frequently used job descriptions
- Toast notification system

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Frontend (Vanilla JS)                 │
│  index.html  ·  script.js  ·  style.css                 │
│  Drag & Drop · JD Library · Dark Mode · Results Table   │
└────────────────────────┬────────────────────────────────┘
                         │  HTTP (FormData)
                         ▼
┌─────────────────────────────────────────────────────────┐
│               FastAPI Backend  (api.py)                  │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │  PDF / DOCX  │  │  AI Router   │  │   Location    │  │
│  │   Parser     │  │  (Gemini /   │  │   Engine      │  │
│  │  pypdf       │  │   OpenAI /   │  │  Nominatim +  │  │
│  │  python-docx │  │   Custom)    │  │  OSRM +       │  │
│  └──────┬───────┘  └──────┬───────┘  │  Haversine    │  │
│         │                 │          └───────┬───────┘  │
│         ▼                 ▼                  ▼          │
│  ┌─────────────────────────────────────────────────┐    │
│  │          Evaluation Pipeline                     │    │
│  │  Deduplication → AI Scoring → Location Check     │    │
│  │  → Qualification Gate → Rank & Return            │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+**
- **Google Gemini API Key** — [Get one here](https://aistudio.google.com/apikey)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Priyanshus2k03/ATS_Engine.git
cd ATS_Engine

# 2. Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install fastapi uvicorn pypdf python-docx python-dotenv \
    google-genai numpy scikit-learn geopy requests pydantic

# 4. Set up environment variables
#    Create a .env file in the project root:
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

### Running the Application

```bash
# Start the FastAPI server
uvicorn api:app --reload
```

The API server starts at `http://127.0.0.1:8000`.

Open `index.html` directly in your browser (or serve it via a static file server) to use the web interface.

> **Tip:** The frontend connects to `http://127.0.0.1:8000` by default. If you change the port, update the `fetch` URL in `script.js`.

---

## 📡 API Reference

### `POST /extract-jd-params/`

Extracts structured requirements from a raw job description using AI.

| Parameter         | Type     | Default    | Description                        |
|-------------------|----------|------------|------------------------------------|
| `job_description` | `string` | *required* | The full job description text      |
| `ai_provider`     | `string` | `"gemini"` | AI engine to use for extraction    |

**Response:**

```json
{
  "min_experience_years": 2.0,
  "max_experience_years": 5.0,
  "required_skills": "Python, React, SQL, Docker, AWS",
  "required_education": "Bachelors",
  "target_location": "Mumbai"
}
```

---

### `POST /analyze-batch-parallel/`

Evaluates a batch of resumes against a job description in parallel.

| Parameter                | Type       | Default             | Description                                      |
|--------------------------|------------|---------------------|--------------------------------------------------|
| `files`                  | `File[]`   | *required*          | Resume files (PDF/DOCX)                          |
| `job_description`        | `string`   | *required*          | The full job description                         |
| `min_experience_years`   | `float`    | `0.0`               | Minimum experience required                      |
| `max_experience_years`   | `float`    | `7.0`               | Maximum experience required                      |
| `mandatory_experience`   | `bool`     | `false`             | Hard-filter on experience range                  |
| `required_skills`        | `string`   | `""`                | Comma-separated list of required skills          |
| `mandatory_skills`       | `bool`     | `false`             | Hard-filter on skills                            |
| `required_education`     | `string`   | `""`                | Required education level                         |
| `mandatory_education`    | `bool`     | `false`             | Hard-filter on education                         |
| `target_location`        | `string`   | `"Borivali, Mumbai"`| Office / target location                         |
| `mandatory_location`     | `bool`     | `false`             | Hard-filter on location (rejects "Relocation")   |
| `passing_score`          | `int`      | `60`                | Minimum total score to qualify                   |
| `shortlist_top_n`        | `int`      | `0`                 | Return top N candidates (0 = return all)         |
| `ai_provider`            | `string`   | `"gemini"`          | AI engine (`gemini` / `openai` / `custom`)       |

**Response:** Array of `CandidateEvaluation` objects, sorted by qualification status and score.

```json
[
  {
    "candidate_name": "Priya Sharma",
    "total_score": 82,
    "experience_score": 35,
    "experience_details": "4 years of relevant full-stack experience.",
    "skills_score": 25,
    "skills_details": "Strong match on Python, React; missing Docker.",
    "education_score": 22,
    "education_details": "B.Tech in Computer Science from a reputed university.",
    "score_justification": "Strong candidate with relevant experience and skills.",
    "candidate_location": "Andheri, Mumbai - ~32 min drive (Excellent Commute)",
    "location_relevancy": "High",
    "location_details": "~32 min drive (Excellent Commute)",
    "contact_email": "priya@example.com",
    "contact_phone": "+91-9876543210",
    "experience_years": 4.0,
    "skills": ["Python", "React", "SQL", "JavaScript"],
    "missing_requirements": ["Docker", "AWS"],
    "is_qualified": true,
    "source_file": "priya_sharma_resume.pdf"
  }
]
```

---

## ⚙️ Configuration

Configuration is managed via constants in [api.py](api.py):

| Variable         | Default               | Description                                           |
|------------------|-----------------------|-------------------------------------------------------|
| `GEMINI_API_KEY` | —                     | Your Google Gemini API key                            |
| `GEMINI_MODEL`   | `gemini-3.1-flash-lite` | The Gemini model variant to use                      |
| `MAX_CONCURRENT` | `2`                   | Max parallel AI requests (increase with caution)      |

### Environment Variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_api_key_here
```

---

## 📂 Project Structure

```
ATS_Engine/
├── api.py           # FastAPI backend — AI evaluation, location engine, API routes
├── index.html       # Frontend — form, results table, candidate modal
├── script.js        # Frontend logic — drag & drop, JD library, API calls, theme
├── style.css        # Styling — light/dark themes, responsive layout, animations
├── env              # Environment variable reference
├── .gitignore       # Git ignore rules
├── README.md        # This file
└── venv/            # Python virtual environment (not committed)
```

---

## 🔌 AI Provider Extensibility

The backend includes a pluggable **AI Router** pattern, making it straightforward to add new LLM providers:

```python
async def call_ai_engine(prompt: str, ai_provider: str) -> dict:
    if ai_provider == "gemini":
        # ✅ Active — Gemini 3.1 Flash Lite
        ...
    elif ai_provider == "openai":
        # 🔜 Placeholder — OpenAI GPT integration
        raise NotImplementedError()
    elif ai_provider == "custom":
        # 🔜 Placeholder — Custom / Local LLM API
        raise NotImplementedError()
```

To add a new provider, implement the corresponding branch in `call_ai_engine()` within [api.py](api.py).

---

## 🧮 Scoring Methodology

| Pillar        | Max Score | Evaluated By       |
|---------------|----------:|--------------------|
| **Experience**|     40    | Years + relevance to JD requirements |
| **Skills**    |     30    | Match against required skill list    |
| **Education** |     30    | Degree level and field alignment     |
| **Location**  |  N/A      | Scored separately via Hybrid Location Engine |

- **Total Score** = Experience + Skills + Education (out of 100)
- Candidates below the **passing threshold** are auto-rejected
- **Mandatory filters** can independently disqualify candidates regardless of score

---

## 📝 License

This project is proprietary to **Muuchstac**. All rights reserved.

---

<p align="center">
  Built with ❤️ using <strong>FastAPI</strong> · <strong>Google Gemini AI</strong> · <strong>Vanilla JS</strong>
</p>
