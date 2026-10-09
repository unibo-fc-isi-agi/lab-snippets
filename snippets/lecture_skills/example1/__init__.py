"""
Example 1: Skills for the Committee.

Two skills, as folders following the Agent Skills specification (https://agentskills.io/specification):
phd-application-review/ (review an application, by the committee's rubric) and admission-letter/ (write the outcome letter).
validate.py checks them, and prints what an agent sees of them at startup.
"""
from pathlib import Path

FOLDER = Path(__file__).parent  # i.e. the folder containing the skills, to be given to agents (or harnesses)
