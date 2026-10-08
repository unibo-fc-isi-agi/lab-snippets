"""
The same letter-scoring system of Example 1, with LangChain: a prompt template, piped into an LLM with structured output.

Run with: poetry run python -m snippets -l prompting -e 1bis mario-rossi
(the letter can be given as a candidate's ID, or as a path, e.g. data/letter-mario-rossi.txt)
Configure via env vars: OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL.
"""
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

base_url = os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1/")
api_key = os.environ.get("OPENAI_API_KEY") or input(f"Enter your API key for {base_url}: ")
model = os.environ.get("OPENAI_MODEL", "openrouter/auto")

llm = ChatOpenAI(base_url=base_url, api_key=api_key, model=model)


# a prompt template: messages with placeholders (e.g. {letter_text}), filled in at invocation time
prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are assisting the admission commettee of University of Bologna PhD programmes.
Your goal is to help assess recommendation letters for PhD applications.
You'll be provided with the text of each letter,
and you'll have to analyse it to extract structured information in JSON.
"""),
    ("user", "Score the following letter:\n\n<letter>\n{letter_text}\n</letter>"),
])

scoring_instructions = """
Maximum score is +5, minimum score is 0.
Start by maximum and decrease by 1 for each issue among the following:
- -1 per each lacking non-optional field,
- not clear how the author knows the applicant,
- vague relationship between the author and the applicant: no storytelling about the applicant,
- skills mentioned in the letter are very generic and not supported by concrete examples,
- no strengths mentioned in the letter or very generic ones, not supported by concrete examples,
- author mentions weaknesses about the applicant explicitly or implicitly,
- letter is not tailored for a specific PhD programme.
"""


# the output schema: same Pydantic classes as in Example 1
from typing import Optional
from pydantic import BaseModel, Field

class ApplicantInfo(BaseModel):
    """Structured information about the applicant, extracted from the letter."""

    name: str = Field(description="Full name of the applicant, first name first, family name last.")
    degree: Optional[str] = Field(description="Degree obtained by the applicant before the application, according to the letter.")
    alma_mater: Optional[str] = Field(description="University where the applicant obtained their degree, according to the letter.")
    attended: list[str] = Field(description="List of courses or teachings the applicant has attended, as mentioned in the letter.")
    skills: list[str] = Field(description="List of skills possessed by the applicant, as mentioned in the letter.")
    strengths: list[str] = Field(description="List of strengths identified for the applicant, as mentioned in the letter, except skills.")
    weaknesses: list[str] = Field(description="List of weaknesses identified for the applicant, as mentioned in the letter.")

class AuthorInfo(BaseModel):
    """Structured information about the author of the letter, extracted from the letter."""

    name: str = Field(description="Full name of the author, first name first, family name last.")
    affiliation: str = Field(description="Affiliation of the author, according to the letter.")
    position: Optional[str] = Field(description="Position of the author, according to the letter, in their institution.")
    seniority: Optional[int] = Field(description="Seniority of the author: 1=researcher or lecturer, 2=assistant professor, 3=associate professor, 4=full professor, 0=other. " \
                                        "Scale similarly in case of non-academic positions, e.g., 1=entry-level, 2=mid-level, 3=senior, 4=executive.")
    email: str = Field(description="Email of the author, according to the letter.")
    nationality: Optional[str] = Field(description="Nationality of the author, according to the letter.")
    relationship_with_applicant: Optional[str] = Field(description="Relationship of the author with the applicant, according to the letter.")

class LetterInfo(BaseModel):
    applicant: ApplicantInfo = Field(description="Information about the person the letter is about, i.e., the one applying for the PhD programme.")
    author: AuthorInfo = Field(description="Information about the person who wrote the letter, i.e., the one recommending the applicant for the PhD programme.")
    application_for: str = Field(description="The PhD programme for which the applicant is applying.")
    score: int = Field(description=f"The score assigned to the letter according to the following rules.{scoring_instructions}")


# a chain: the filled-in prompt goes to the LLM, whose answer is parsed into a LetterInfo
scoring_chain = prompt | llm.with_structured_output(LetterInfo)

def score_letter(letter_text: str) -> LetterInfo:
    return scoring_chain.invoke({"letter_text": letter_text})  # fills the {letter_text} placeholder


if __name__ == "__main__":
    import sys
    import data

    print(f"Using model: {model}")

    letter_path = sys.argv[1] if len(sys.argv) > 1 else input("Enter the letter to score (candidate's ID or path): ")
    letter_path = data.find_letter(letter_path)
    if not letter_path.is_file():
        print(f"File {letter_path} is not a file.")
        sys.exit(1)
    letter_text = letter_path.read_text()
    letter_info = score_letter(letter_text)
    print(letter_info.model_dump_json(indent=2))
