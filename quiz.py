"""Core quiz logic: turns lecture notes into validated multiple-choice questions."""

import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = "gpt-4o-mini"  # check OpenAI's models page for a current, cheap model
MAX_NOTES_CHARS = 6000
NUM_QUESTIONS = 5


class QuizError(Exception):
    """Raised when notes are invalid or the model returns unusable output."""


def build_prompt(notes: str) -> str:
    """Build the instruction we send to the model.

    Args:
        notes: the lecture notes pasted by the user.
    Returns:
        The full prompt string.
    """
    return f"""Create {NUM_QUESTIONS} multiple-choice questions based ONLY on the notes below.

Return JSON in exactly this shape:
{{"questions": [
  {{"question": "...", "options": ["A", "B", "C", "D"], "answer": 0, "explanation": "..."}}
]}}

Rules:
- Each question has exactly 4 options.
- "answer" is the index (0-3) of the correct option.
- Only use information found in the notes.

NOTES:
{notes}"""


def validate_questions(data: dict) -> list[dict]:
    """Check the model's JSON has the structure we expect.

    Args:
        data: parsed JSON from the model.
    Returns:
        The list of question dicts if valid.
    Raises:
        QuizError: if anything is missing or the wrong type.
    """
    questions = data.get("questions")
    if not isinstance(questions, list) or not questions:
        raise QuizError("No questions returned.")

    for q in questions:
        if not isinstance(q.get("question"), str):
            raise QuizError("A question is missing its text.")
        options = q.get("options")
        if not isinstance(options, list) or len(options) != 4:
            raise QuizError("Each question needs exactly 4 options.")
        if q.get("answer") not in (0, 1, 2, 3):
            raise QuizError("Answer must be an index from 0 to 3.")
        if not isinstance(q.get("explanation"), str):
            raise QuizError("A question is missing its explanation.")
    return questions


def generate_questions(notes: str, retries: int = 1) -> list[dict]:
    """Generate a validated quiz from lecture notes.

    Args:
        notes: lecture notes as plain text.
        retries: how many times to retry if the model returns bad output.
    Returns:
        A list of question dicts (question, options, answer, explanation).
    Raises:
        QuizError: if the notes are invalid or every attempt fails.
    """
    notes = notes.strip()
    if not notes:
        raise QuizError("Notes are empty.")
    if len(notes) > MAX_NOTES_CHARS:
        raise QuizError(f"Notes are too long (max {MAX_NOTES_CHARS} characters).")

    last_error = None
    for _ in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": build_prompt(notes)}],
                response_format={"type": "json_object"},  # forces valid JSON
            )
            data = json.loads(response.choices[0].message.content)
            return validate_questions(data)
        except (json.JSONDecodeError, QuizError) as e:
            last_error = e  # bad output, so try again
    raise QuizError(f"Could not generate a valid quiz: {last_error}")


if __name__ == "__main__":
    sample = """A list in Python is an ordered, mutable collection. Items are
    accessed by index starting at 0. Use append() to add an item to the end,
    and pop() to remove and return an item. Lists can hold mixed types."""
    for q in generate_questions(sample):
        print(q["question"])
        for i, option in enumerate(q["options"]):
            print(f"  {i}. {option}")
        print("  Answer:", q["answer"], "-", q["explanation"], "\n")