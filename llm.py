"""
Movie recommender for the TMDB top movies dataset.

Implements:
    get_recommendation(preferences: str, history: list[str], history_ids: list[int] = []) -> dict

Returns:
    {
        "tmdb_id": <int>,
        "description": <str>
    }
"""

import argparse
import json
import math
import os
import queue
import re
import threading
import time

import ollama
import pandas as pd


MODEL = "gemma4:31b-cloud"
DATA_PATH = os.path.join(os.path.dirname(__file__), "tmdb_top1000_movies.csv")

SHORTLIST_SIZE = 8
MIN_VOTE_COUNT = 500
LLM_TIMEOUT_SECONDS = 18.0

MOVIES = pd.read_csv(DATA_PATH).copy()

for col in ["title", "genres", "overview", "keywords"]:
    if col in MOVIES.columns:
        MOVIES[col] = MOVIES[col].fillna("").astype(str)

MOVIES["tmdb_id"] = pd.to_numeric(MOVIES["tmdb_id"], errors="coerce").fillna(-1).astype(int)
MOVIES["vote_average"] = pd.to_numeric(MOVIES["vote_average"], errors="coerce").fillna(0.0)
MOVIES["vote_count"] = pd.to_numeric(MOVIES["vote_count"], errors="coerce").fillna(0)
MOVIES["year"] = MOVIES["year"].fillna("").astype(str)

STOPWORDS = {
    "i", "like", "want", "with", "and", "the", "a", "an", "to", "of", "for",
    "movie", "movies", "something", "that", "is", "are", "it", "me", "my",
    "watch", "looking", "love", "enjoy", "really", "very", "film", "films"
}

GENRE_TRIGGERS = {
    "animated": "animation",
    "animation": "animation",
    "disney": "animation",
    "cartoon": "animation",
    "horror": "horror",
    "scary": "horror",
    "comedy": "comedy",
    "funny": "comedy",
    "rom com": "romance",
    "romcom": "romance",
    "romantic": "romance",
    "romance": "romance",
    "documentary": "documentary",
    "action": "action",
    "thriller": "thriller",
    "sci-fi": "science fiction",
    "scifi": "science fiction",
    "musical": "music",
    "musicals": "music",
    "music": "music",
    "family": "family",
    "adventure": "adventure",
    "fantasy": "fantasy",
    "drama": "drama",
    "crime": "crime",
    "mystery": "mystery",
}

# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    text = str(text).lower().strip()
    text = text.replace("&", " and ")
    text = text.replace("sci-fi", "science fiction")
    text = text.replace("scifi", "science fiction")
    text = text.replace("rom-com", "romance comedy")
    text = text.replace("romcom", "romance comedy")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def simple_stem(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("s"):
        return word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    return [
        simple_stem(w)
        for w in normalize_text(text).split()
        if len(w) >= 3 and w not in STOPWORDS
    ]


def split_genres(text: str) -> set[str]:
    tokens = set()
    for part in re.split(r"[|,/]", str(text)):
        for word in normalize_text(part).split():
            word = simple_stem(word)
            if len(word) >= 3 and word not in STOPWORDS:
                tokens.add(word)
    return tokens

# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------
def get_preferred_lang_code(preferences: str) -> str | None:
    pref_norm = normalize_text(preferences)
    lang_map = {
        "japanese": "ja", "japan": "ja",
        "korean": "ko", "korea": "ko",
        "chinese": "zh", "china": "zh",
        "french": "fr", "france": "fr",
        "spanish": "es", "spain": "es",
    }
    for keyword, code in lang_map.items():
        if keyword in pref_norm:
            return code
    return None


# ---------------------------------------------------------------------------
# Genre filtering
# ---------------------------------------------------------------------------

def detect_required_genres(preferences: str) -> list[str]:
    pref_lower = preferences.lower()
    required = []
    for trigger, genre in GENRE_TRIGGERS.items():
        if trigger in pref_lower and genre not in required:
            required.append(genre)
    return required


# ---------------------------------------------------------------------------
# History utilities
# ---------------------------------------------------------------------------

def title_matches_history(movie_title: str, history_item: str) -> bool:
    movie_norm = normalize_text(movie_title)
    hist_norm = normalize_text(history_item)
    if not movie_norm or not hist_norm:
        return False
    if movie_norm == hist_norm:
        return True
    if hist_norm in movie_norm or movie_norm in hist_norm:
        return True
    movie_compact = movie_norm.replace(" ", "")
    hist_compact = hist_norm.replace(" ", "")
    if movie_compact == hist_compact:
        return True
    if hist_compact in movie_compact or movie_compact in hist_compact:
        return True
    movie_tokens = set(tokenize(movie_norm))
    hist_tokens = set(tokenize(hist_norm))
    if hist_tokens and len(hist_tokens) >= 2 and hist_tokens.issubset(movie_tokens):
        return True
    if len(hist_tokens) >= 3 and len(movie_tokens & hist_tokens) >= len(hist_tokens) - 1:
        return True
    if hist_norm.startswith("the "):
        root = hist_norm[4:].strip()
        if root and root in movie_norm:
            return True
    return False

# ---------------------------------------------------------------------------
# Candidate filtering and scoring
# ---------------------------------------------------------------------------

def build_shortlist(preferences: str, history: list[str], history_ids: list[int]) -> pd.DataFrame:
    df = MOVIES.copy()

    # Exclude watched by ID
    valid_ids = []
    for x in history_ids:
        try:
            valid_ids.append(int(x))
        except Exception:
            pass
    if valid_ids:
        df = df[~df["tmdb_id"].isin(valid_ids)]

    # Exclude watched by title
    clean_history = [h.strip() for h in history if str(h).strip()]
    if clean_history:
        seen_mask = df["title"].apply(
            lambda title: any(title_matches_history(str(title), h) for h in clean_history)
        )
        df = df[~seen_mask]

    # Minimum popularity filter
    popular = df[df["vote_count"] >= MIN_VOTE_COUNT].copy()
    if not popular.empty:
        df = popular

    # Language filter
    preferred_lang_code = get_preferred_lang_code(preferences)
    if preferred_lang_code:
        df_lang = df[df["original_language"] == preferred_lang_code].copy()
        if len(df_lang) >= 3:
            df = df_lang

    if df.empty:
        df = MOVIES.copy()

    # Scoring
    pref_tokens = set(tokenize(preferences))
    required_genres = detect_required_genres(preferences)

    df["genre_match"] = df["genres"].apply(lambda x: len(pref_tokens & split_genres(x)))
    df["title_match"] = df["title"].apply(lambda x: len(pref_tokens & set(tokenize(x))))
    df["overview_match"] = df["overview"].apply(lambda x: len(pref_tokens & set(tokenize(x))))
    df["keyword_match"] = df["keywords"].apply(lambda x: len(pref_tokens & set(tokenize(x))))
    df["required_genre_bonus"] = df["genres"].apply(
        lambda x: sum(1 for genre in required_genres if genre in str(x).lower())
    )

    df["score"] = (
        4.0 * df["genre_match"]
        + 2.0 * df["title_match"]
        + 3.0 * df["overview_match"]
        + 1.5 * df["keyword_match"]
        + 1.5 * df["required_genre_bonus"]
        + 0.2 * df["vote_average"]
        + 0.03 * df["vote_count"].apply(lambda x: math.log1p(max(x, 0)))
    )

    return df.sort_values(
        by=["score","vote_average","vote_count"],
        ascending=[False, False, False]
    ).head(SHORTLIST_SIZE).reset_index(drop=True)

# ---------------------------------------------------------------------------
# Fallback description (no LLM needed)
# ---------------------------------------------------------------------------

def make_local_description(preferences: str, row) -> str:
    title = str(row["title"]).strip()
    genres = str(row.get("genres", "")).strip()
    overview = re.sub(r"\s+", " ", str(row.get("overview", ""))).strip()

    pref_tokens = set(tokenize(preferences))
    genre_tokens = split_genres(genres)
    matched = list(pref_tokens & genre_tokens)

    if matched:
        if len(matched) == 1:
            opening = (
                f"{title} is a strong match if you're in the mood for "
                f"a {matched[0]} movie"
            )
        else:
            genre_phrase = " and ".join(matched[:2])
            opening = (
                f"{title} is a strong match if you want a movie with "
                f"{genre_phrase} elements"
            )
    elif genres:
        opening = f"{title} stands out for its {genres.lower()} elements"
    else:
        opening = f"{title} is an easy recommendation for your taste"

    second = ""
    if overview:
        clean_overview = overview.strip()
    # Avoid cutting common abbreviations such as L.A. or U.S.
    protected = (
        clean_overview
        .replace("L.A.", "LA")
        .replace("U.S.", "US")
        .replace("U.K.", "UK")
    )

    sentences = re.split(r"(?<=[.!?])\s+", protected, maxsplit=1)
    first_sentence = sentences[0].strip()

    first_sentence = (
        first_sentence
        .replace("LA", "L.A.")
        .replace("US", "U.S.")
        .replace("UK", "U.K.")
    )

    if first_sentence:
        second = f" {first_sentence}"
    

    closing = " It has the tone and story to fit what you're looking for."

    description = opening + "." + second + closing
    return re.sub(r"\s+", " ", description).strip()[:500]


# ---------------------------------------------------------------------------
# LLM call with timeout
# ---------------------------------------------------------------------------

def call_llm(prompt: str) -> dict:
    api_key = os.getenv("OLLAMA_API_KEY")
    if not api_key:
        raise RuntimeError("OLLAMA_API_KEY is not set")

    client = ollama.Client(
        host="https://ollama.com",
        headers={"Authorization": f"Bearer {api_key}"},
    )

    response = client.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        format="json",
        options={"num_predict": 180},
    )

    content = response.message.content.strip()
    content = re.sub(r'^```json\s*', '', content)
    content = re.sub(r'\s*```$', '', content)
    content = content.strip()

    try:
        return json.loads(content)
    except Exception:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            raise ValueError("Model did not return valid JSON")
        return json.loads(match.group(0))


def call_llm_with_timeout(prompt: str, timeout_seconds: float) -> dict:
    q = queue.Queue()

    def worker():
        try:
            q.put(("ok", call_llm(prompt)))
        except Exception as e:
            q.put(("error", e))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout_seconds)

    if thread.is_alive():
        raise TimeoutError(f"LLM call exceeded {timeout_seconds} seconds")
    if q.empty():
        raise RuntimeError("LLM worker returned no result")

    status, payload = q.get()
    if status == "ok":
        return payload
    raise payload


# ---------------------------------------------------------------------------
# Main recommendation function
# ---------------------------------------------------------------------------

def get_recommendation(
    preferences: str,
    history: list[str] | None = None,
    history_ids: list[int] | None = None
) -> dict:
    """Return a dict with keys 'tmdb_id' (int) and 'description' (str)."""

    history = history or []
    history_ids = history_ids or []
    """Return a dict with keys 'tmdb_id' (int) and 'description' (str)."""

    shortlist = build_shortlist(preferences, history, history_ids)
    if shortlist.empty:
        shortlist = MOVIES.nlargest(SHORTLIST_SIZE, "vote_average").reset_index(drop=True)

    best = shortlist.iloc[0]

    valid_ids = {int(x) for x in shortlist["tmdb_id"].tolist()}
    valid_ids_str = ", ".join(str(x) for x in valid_ids)

    movie_list = "\n".join(
        f'- tmdb_id={int(row.tmdb_id)} | "{row.title}" ({row.year}) | {row.vote_average}/10 | '
        f'genres: {row.genres} | overview: {row.overview[:160]}'
        for row in shortlist.itertuples()
    )

    history_text = (
        ", ".join(f'"{name}"' for name in history) if history else "none"
    )

    watched_movies = MOVIES[MOVIES["tmdb_id"].isin(history_ids)]
    if not watched_movies.empty:
        watched_genres = watched_movies["genres"].fillna("").str.cat(sep=", ")
        history_insight = f"User tends to enjoy: {watched_genres[:100]}"
    else:
        history_insight = ""

    prompt = f"""You are a movie recommendation assistant.

User preferences: "{preferences}"
Already watched (do NOT recommend): {history_text}
{history_insight}

Candidates:
{movie_list}

Allowed tmdb_id values: [{valid_ids_str}]

Pick the single best match. Respond ONLY with this JSON, no markdown, no code fences:
{{
  "tmdb_id": <integer from the allowed list>,
  "description": "<a natural, engaging recommendation of 2 to 3 sentences, around 180 to 320 characters, under 500 characters total>"
}}

Rules:
- Pick only from the candidate list above
- The tmdb_id MUST be one of the allowed values
- Do not recommend anything already watched
- Avoid spoilers
- Start the description with the movie title
- Keep the description under 500 characters
- Write in a natural, engaging recommendation style
- Explain clearly why the movie fits the user's preferences
- Write 2-3 complete sentences
- Use some of the exact words the user typed in their preference to make it feel personalized
- Do not copy the overview directly; rephrase it into a more appealing recommendation
"""

    try:
        result = call_llm_with_timeout(prompt, LLM_TIMEOUT_SECONDS)

        chosen_id = int(result.get("tmdb_id", -1))
        description = str(result.get("description", "")).strip()

        # Validate the returned ID is allowed
        if chosen_id not in valid_ids:
            return {
                "tmdb_id": int(best["tmdb_id"]),
                "description": make_local_description(preferences, best),
            }

        # Double-check it's not in watch history
        if chosen_id in {int(x) for x in history_ids}:
            return {
                "tmdb_id": int(best["tmdb_id"]),
                "description": make_local_description(preferences, best),
            }

        if not description:
            chosen_row = shortlist[shortlist["tmdb_id"] == chosen_id].iloc[0]
            description = make_local_description(preferences, chosen_row)

        return {
            "tmdb_id": chosen_id,
            "description": re.sub(r"\s+", " ", description).strip()[:500],
        }

    except Exception:
        return {
            "tmdb_id": int(best["tmdb_id"]),
            "description": make_local_description(preferences, best),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a local movie recommendation test.")
    parser.add_argument("--preferences", type=str)
    parser.add_argument("--history", type=str)
    args = parser.parse_args()

    print("Movie recommender - type your preferences and press Enter.")

    preferences = (
        args.preferences.strip()
        if args.preferences and args.preferences.strip()
        else input("Preferences: ").strip()
    )
    history_raw = (
        args.history.strip()
        if args.history and args.history.strip()
        else input("Watch history (optional): ").strip()
    )
    history = [t.strip() for t in history_raw.split(",") if t.strip()] if history_raw else []

    print("\nThinking...\n")
    start = time.perf_counter()
    result = get_recommendation(preferences, history)
    print(result)
    elapsed = time.perf_counter() - start
    print(f"\nServed in {elapsed:.2f}s")