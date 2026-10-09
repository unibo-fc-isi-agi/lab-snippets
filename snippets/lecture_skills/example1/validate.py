"""
Validates the skills in the given folders (default: the ones of this example), and prints what an agent sees of them at startup.

Run with: poetry run python -m snippets -l skills -e 1 [FOLDER ...]
"""
import sys
from snippets.lecture_skills.example1 import FOLDER
from snippets.lecture_skills.skills import catalogue, discover, validate

skills = discover(*(sys.argv[1:] or [FOLDER]))
valid = True
for skill in skills.values():
    errors = validate(skill)
    valid = valid and not errors
    print("✓" if not errors else "✗", skill.name, f"({skill.path})")
    for error in errors:
        print("    -", error)

print("\n# What the agent sees at startup (level 1 of progressive disclosure):")
print(catalogue(skills) or "(no skills found)")
sys.exit(0 if valid and skills else 1)
