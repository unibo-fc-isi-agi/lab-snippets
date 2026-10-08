"""
The test dataset of the letter-scoring system, shared by the examples of this lecture.
Test cases are loaded from test_data.yml (next to this file), each one with an "input" (a letter file name)
and its "expectations" (the results expected from a correct scoring).
"""
from pathlib import Path
import yaml
import data

TEST_DATA_FILE = Path(__file__).parent / "test_data.yml"

# list of test cases, e.g. [{"input": "letter-mario-rossi.txt", "expectations": {"applicant_name": "Mario Rossi", ...}}, ...]
TEST_CASES = yaml.safe_load(TEST_DATA_FILE.read_text())


def read_letter(case: dict) -> str:
    return (data.DIR / case["input"]).read_text()  # input files are looked up in <project root>/data/
