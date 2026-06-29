# LLM Movie Recommender

A hybrid movie recommendation system built on the TMDB Top 1000 movies dataset.

The system takes a user's free-text preferences and watch history, filters and ranks candidate movies, and returns one movie recommendation with a short explanation. It combines rule-based filtering, content-based scoring, and optional LLM-based final selection.

## Features

- Filters out movies the user has already watched
- Supports both watch-history titles and TMDB IDs
- Uses fuzzy title matching for messy inputs such as partial titles or different punctuation
- Scores movies using genre, title, overview, keywords, vote average, and vote count
- Detects explicit language preferences such as Japanese, Korean, Chinese, French, and Spanish
- Uses an LLM to select from a shortlist and generate a natural recommendation
- Falls back to a local rule-based description if the LLM is unavailable, slow, or returns invalid output
- Validates the LLM-selected `tmdb_id` to reduce hallucinated recommendations

## Project Structure

```text
llm-movie-recommender/
├── llm.py
├── tmdb_top1000_movies.csv
├── requirements.txt
└── README.md
```

## How It Works

The recommender uses a hybrid pipeline:

1. Normalize the user preference text and movie metadata.
2. Remove watched movies using TMDB IDs and fuzzy title matching.
3. Apply a minimum popularity filter.
4. Apply explicit language filtering when the user requests a specific language.
5. Score candidate movies based on genre, title, overview, keywords, rating, and popularity.
6. Build a shortlist of top candidates.
7. Ask the LLM to choose one movie from the shortlist and generate a short explanation.
8. Validate the LLM response.
9. Use a local fallback recommendation if the LLM fails, times out, or returns an invalid movie ID.

This design keeps the LLM constrained to a relevant candidate set instead of letting it choose freely from the entire dataset.

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

The project expects the dataset file to be named:

```text
tmdb_top1000_movies.csv
```

Place it in the same folder as `llm.py`.

## Environment Variables

The LLM call uses Ollama Cloud. Create a `.env` file or set environment variables manually:

```bash
OLLAMA_API_KEY=your_ollama_api_key_here
OLLAMA_MODEL=gemma4:31b-cloud
```

`OLLAMA_MODEL` is optional. If it is not set, the code uses `gemma4:31b-cloud` by default.

If no API key is provided, the project still runs by using the local fallback description generator.

## Usage

Interactive mode:

```bash
python llm.py
```

Example command-line run:

```bash
python llm.py --preferences "dark crime thriller" --history "The Dark Knight, Joker"
```

Example output:

```json
{
  "tmdb_id": 242582,
  "description": "Nightcrawler is a strong match if you want a movie with crime and thriller elements. When Lou Bloom, desperate for work, muscles into the world of L.A. crime journalism, he blurs the line between observer and participant to become the star of his own story. It has the tone and story to fit what you're looking for."
}
```

You can also pass watched movie IDs:

```bash
python llm.py --preferences "Japanese animation" --history "Spirited Away" --history-ids "129"
```

## Main Components

- `normalize_text()`  
  Standardizes user input and movie text.

- `title_matches_history()`  
  Performs fuzzy matching so watched movies are not recommended again.

- `build_shortlist()`  
  Filters, scores, and ranks candidate movies.

- `call_llm()`  
  Sends the shortlist to the LLM and parses the JSON response.

- `call_llm_with_timeout()`  
  Prevents slow LLM calls from blocking the system.

- `make_local_description()`  
  Generates a local fallback explanation when the LLM cannot be used.

- `get_recommendation()`  
  Runs the full recommendation pipeline and returns the final result.

## Robustness Checks

The system includes several safeguards:

- Excludes watched movies before scoring
- Validates that the selected `tmdb_id` is in the shortlist
- Rejects invalid LLM-selected movie IDs
- Handles invalid JSON responses from the LLM
- Uses a timeout for LLM calls
- Falls back to deterministic local output when needed

## Limitations

- Broad prompts may still produce similar high-popularity recommendations.
- Fuzzy history matching may not catch every franchise-level duplicate.
- Language filtering works best when the user explicitly names a language.
- The local fallback description is simpler than the LLM-generated explanation.

## Tech Stack

- Python
- pandas
- Ollama
