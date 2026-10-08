"""
Offline tests of the exercises' solutions: they check the deterministic parts (caching, retries, scoring, voting, validation,
approvals, gateway policies, ...), with NO call to any LLM, so they run in CI. Run with: poetry run poe test
"""
import os

os.environ.setdefault("OPENAI_API_KEY", "dummy")  # snippets read it upon import (and would ask for it interactively, if missing)
