"""
Classifies the tone of a recommendation letter (enthusiastic, lukewarm, critical) with several prompting techniques:
zero-shot, few-shot, chain-of-thought, and self-consistency (i.e. majority vote over several samples).

Run with: poetry run python -m snippets -l prompting -e 2 {zero-shot|few-shot|cot} LETTER [SAMPLES]
e.g.:     poetry run python -m snippets -l prompting -e 2 cot jean-dupont 5
Configure via env vars: OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL.
"""
import os
from collections import Counter
from typing import Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

base_url = os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1/")
api_key = os.environ.get("OPENAI_API_KEY") or input(f"Enter your API key for {base_url}: ")
model = os.environ.get("OPENAI_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")

llm = ChatOpenAI(base_url=base_url, api_key=api_key, model=model, temperature=1.0)  # some randomness, for self-consistency

Tone = Literal["enthusiastic", "lukewarm", "critical"]  # the only admissible labels

system = ("system", """You classify the tone of recommendation letters for PhD applications.
Use exactly one label among: enthusiastic, lukewarm, critical.
- enthusiastic: strong, specific, and unreserved support
- lukewarm: positive, but generic or shallow, with little evidence
- critical: explicit or implicit reservations about the applicant""")
user = ("user", "<letter>\n{letter}\n</letter>")

# 1. zero-shot: instructions only
zero_shot = ChatPromptTemplate.from_messages([system, user])

# 2. few-shot: instructions + solved examples, rendered as past conversation turns
examples = [
    {"letter": "Anna led our lab's move to Rust, and her thesis on lock-free queues became a paper at a top venue. "
               "She is the best student I supervised in ten years.", "label": "enthusiastic"},
    {"letter": "Luca attended my course on distributed systems and passed the exam. "
               "He is a hard-working and motivated student, and I recommend him.", "label": "lukewarm"},
    {"letter": "Sara is smart, although she sometimes needed reminders to meet deadlines. "
               "With proper supervision, she could do well in a PhD.", "label": "critical"},
]
few_shot = ChatPromptTemplate.from_messages([
    system,
    FewShotChatMessagePromptTemplate(
        examples=examples,
        example_prompt=ChatPromptTemplate.from_messages([user, ("ai", "{label}")]),
    ),
    user,
])

class ToneOnly(BaseModel):
    label: Tone = Field(description="The tone of the letter")

# 3. chain-of-thought: the model must write its reasoning *before* the label (fields are generated in order)
class ToneWithReasoning(BaseModel):
    reasoning: str = Field(description="Step by step: quote the key sentences of the letter, and discuss what they reveal about the author's opinion of the applicant")
    label: Tone = Field(description="The tone of the letter, decided according to the reasoning above")

# one chain per technique: prompt template | LLM with structured output
chains = {
    "zero-shot": zero_shot | llm.with_structured_output(ToneOnly),
    "few-shot": few_shot | llm.with_structured_output(ToneOnly),
    "cot": zero_shot | llm.with_structured_output(ToneWithReasoning),
}

# 4. self-consistency: sample several answers (in parallel), then take a majority vote
def classify(letter: str, technique: str, samples: int = 1) -> tuple[Tone, Counter]:
    answers = chains[technique].batch([{"letter": letter}] * samples)
    votes = Counter(answer.label for answer in answers)
    return votes.most_common(1)[0][0], votes


if __name__ == "__main__":
    import sys
    import data

    if len(sys.argv) < 3 or sys.argv[1] not in chains:
        sys.exit(f"usage: {sys.argv[0]} {{{'|'.join(chains)}}} LETTER [SAMPLES]")
    technique, letter = sys.argv[1], data.find_letter(sys.argv[2]).read_text()
    samples = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    print(f"Using model: {model}")
    for message in chains[technique].first.format_messages(letter="..."):  # show the prompt template
        print(message.pretty_repr())
    label, votes = classify(letter, technique, samples)
    print(f"Tone: {label} (votes: {dict(votes)})")
