"""
Extracts structured information from the picture of an ID document, via a vision LLM,
querying it several times and keeping, field by field, the most voted value (self-consistency);
fields with no clear majority are flagged for human review.

Run with: poetry run python -m snippets -l prompting -x 2 mario-rossi [SAMPLES]
(the document can be given as a candidate's ID, i.e. their passport, or as the path of a picture)
Configure via env vars: OPENAI_BASE_URL, OPENAI_API_KEY, VISION_MODEL (must support images, default: google/gemma-4-26b-a4b-it:free).
"""
import base64
import mimetypes
import os
from collections import Counter
from datetime import date
from pathlib import Path
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

base_url = os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1/")
api_key = os.environ.get("OPENAI_API_KEY") or input(f"Enter your API key for {base_url}: ")
# step 1: a model with vision capabilities (a separate env var, as other snippets may need a model with other capabilities)
vision_model = os.environ.get("VISION_MODEL", "google/gemma-4-26b-a4b-it:free")

vision_llm = ChatOpenAI(base_url=base_url, api_key=api_key, model=vision_model, temperature=1.0)  # some randomness, for voting


# step 3: the output schema (dates as `date`, so that pydantic validates them, and code can compare them)
class IDDocumentInfo(BaseModel):
    """Structured information extracted from an ID document."""

    name: str = Field(description="The full name of the holder, as written in the document: given names first, surname last.")
    nationality: str = Field(description="The nationality of the holder, as written in the document.")
    date_of_birth: date = Field(description="The date of birth of the holder.")
    id_number: str = Field(description="The number of the document, as written in its visual zone (not in the machine-readable zone).")
    expiration_date: date = Field(description="The date of expiry of the document.")


# step 2: system prompt (context, general instructions) + user prompt (specific instructions + the picture)
SYSTEM_PROMPT = """You are an assistant of the admission committee of a PhD programme.
You extract structured information from the pictures of the candidates' ID documents.
Report values exactly as written in the document; documents may be multilingual: prefer the Latin-alphabet version of each value."""

def message_with_image(text: str, image_path: Path) -> HumanMessage:
    """A user message made of a text and a picture (base64-encoded), cf. the slides."""
    mime_type = mimetypes.guess_type(image_path)[0] or "image/png"  # e.g. "image/png" for .png files
    data = base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")
    return HumanMessage(content=[
        {"type": "text", "text": text},
        {"type": "image", "base64": data, "mime_type": mime_type},
    ])

extractor = vision_llm.with_structured_output(IDDocumentInfo)


def sample(image_path: Path, samples: int = 5) -> list[IDDocumentInfo]:
    """Step 4: queries the model `samples` times (in parallel) about the same picture."""
    messages = [SystemMessage(SYSTEM_PROMPT), message_with_image("Please extract the information from this ID document.", image_path)]
    results = extractor.batch([messages] * samples, return_exceptions=True)  # a failed sample (e.g. invalid JSON) is not fatal...
    valid = [r for r in results if isinstance(r, IDDocumentInfo)]
    if not valid:  # ... unless all of them fail
        raise RuntimeError(f"All {samples} samples failed, e.g.: {results[0]}")
    return valid


def vote(results: list[IDDocumentInfo]) -> tuple[IDDocumentInfo, dict[str, float]]:
    """Step 5: for each field, the most voted value, and its agreement rate (fraction of results with that value)."""
    voted, agreement = {}, {}
    for field in IDDocumentInfo.model_fields:
        value, count = Counter(getattr(r, field) for r in results).most_common(1)[0]
        voted[field], agreement[field] = value, count / len(results)
    return IDDocumentInfo(**voted), agreement


def flagged(agreement: dict[str, float]) -> list[str]:
    """Fields with no clear (i.e. absolute) majority: a human should check them."""
    return [field for field, rate in agreement.items() if rate <= 0.5]


def extract(image_path: Path, samples: int = 5) -> tuple[IDDocumentInfo, list[str]]:
    """The whole pipeline: sample, vote, flag. Returns the voted information, and the fields to be reviewed by a human."""
    info, agreement = vote(sample(image_path, samples))
    return info, flagged(agreement)


if __name__ == "__main__":
    import sys
    import data

    # step 6: try it with different documents, e.g. all of the candidates' passports
    document = sys.argv[1] if len(sys.argv) > 1 else input("Enter the document (candidate's ID or path): ")
    path = data.passport(document) if document in data.CANDIDATES else Path(document)
    samples = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    print(f"Using model: {vision_model}, {samples} samples")

    results = sample(path, samples)
    info, agreement = vote(results)
    for field, rate in agreement.items():
        print(f"{field:>16}: {str(getattr(info, field)):<24} (agreement: {rate:.0%}){'  <- REVIEW!' if field in flagged(agreement) else ''}")
