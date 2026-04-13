"""
Bitcoin Future Sentiment Analyzer
Uses a GEPA-optimized prompt + Gemini API to score forward-looking sentiment 1-5.

    1 = Strongly Bearish
    2 = Mildly Bearish
    3 = Neutral
    4 = Mildly Bullish
    5 = Strongly Bullish
"""

import google.generativeai as genai
import argparse
import json
import os


# --- Configuration ---
GEMINI_MODEL = "gemini-1.5-flash"
GEPA_RESULT_PATH = "gepa_result.json"

SCORE_LABELS = {
    1: "Strongly Bearish",
    2: "Mildly Bearish",
    3: "Neutral",
    4: "Mildly Bullish",
    5: "Strongly Bullish",
}


def load_gepa_prompt(gepa_result_path: str = GEPA_RESULT_PATH) -> str:
    """Load the GEPA-optimized system prompt from the saved result file."""
    if not os.path.exists(gepa_result_path):
        raise FileNotFoundError(
            f"No GEPA result found at '{gepa_result_path}'. "
            "Run gepa_optimize.py first."
        )

    with open(gepa_result_path, "r") as f:
        result = json.load(f)

    prompt = result.get("best_candidate", {}).get("system_prompt")
    if not prompt:
        raise ValueError("GEPA result file found but 'best_candidate.system_prompt' is missing.")

    return prompt


def analyze_sentiment(article_text: str, system_prompt: str) -> int:
    """
    Send a news article to Gemini and return a sentiment score 1-5.

    Args:
        article_text: The full text of the news article
        system_prompt: The GEPA-optimized system prompt

    Returns:
        Integer score from 1 (strongly bearish) to 5 (strongly bullish)
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=system_prompt
    )

    user_prompt = (
        f"Analyze the following Bitcoin news article and respond with only "
        f"a JSON object in the format {{\"score\": <integer 1-5>}}.\n\n"
        f"{article_text}"
    )

    response = model.generate_content(
        user_prompt,
        generation_config=genai.types.GenerationConfig(temperature=0.1),
    )

    raw = response.text.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    result = json.loads(raw)
    score = result["score"]

    if score not in SCORE_LABELS:
        raise ValueError(f"Unexpected score: {score}. Expected 1-5.")

    return score


def main():
    parser = argparse.ArgumentParser(
        description="Score forward-looking Bitcoin sentiment 1-5 using Gemini."
    )
    parser.add_argument("--article", type=str, help="Path to a .txt article file")
    parser.add_argument("--text", type=str, help="Article text as inline string")
    parser.add_argument("--gepa-result", type=str, default=GEPA_RESULT_PATH)
    args = parser.parse_args()

    if args.article:
        with open(args.article, "r") as f:
            article_text = f.read()
    elif args.text:
        article_text = args.text
    else:
        raise ValueError("Provide either --article or --text.")

    system_prompt = load_gepa_prompt(args.gepa_result)
    score = analyze_sentiment(article_text, system_prompt)

    print(f"{score} — {SCORE_LABELS[score]}")


if __name__ == "__main__":
    main()
