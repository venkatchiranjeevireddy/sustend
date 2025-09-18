# Call Transcript Analyzer (Flask + Groq)

A small Flask app that:
- Accepts a customer call transcript (UI or JSON API)
- Uses Groq API to summarize (2–3 sentences) and classify sentiment (Positive/Neutral/Negative)
- Prints results to console and appends to `call_analysis.csv` with timestamps
- Offers history page and CSV download

## Setup

1) Create and activate a virtual environment (recommended)
```bash
python3 -m venv .venv
source .venv/bin/activate
```

2) Install dependencies
```bash
pip install -r requirements.txt
```

3) Set your Groq API key
```bash
export GROQ_API_KEY="your_key_here"
# or create a .env file next to app.py with:
# GROQ_API_KEY=your_key_here
```

## Run
```bash
python app.py
# App runs on http://localhost:5000
```

## Endpoints
- Web UI: `/` — paste transcript and analyze
- JSON API: `POST /api/analyze` with body `{ "transcript": "..." }`
- History Table: `/history`
- Download CSV: `/download`

## Example JSON call
```bash
curl -s -X POST http://localhost:5000/api/analyze \
  -H 'Content-Type: application/json' \
  -d '{"transcript": "Hi, I was trying to book a slot yesterday but the payment failed..."}'
```

## Video demo (suggested talking points)
- What the app does and quick architecture overview
- Walkthrough of `analyze_transcript()` and the two Groq prompts
- Live run: paste example, show results and timestamp
- Show `/history` and download `call_analysis.csv`
- Mention error handling for missing API key and request timeouts

## Notes
- Sentiment prompt returns one word for clean UI badges
- Timestamps are UTC for consistency
