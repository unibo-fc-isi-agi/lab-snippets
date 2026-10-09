"""
Discovery, parsing, and validation of skills, following the Agent Skills specification (https://agentskills.io/specification):
a skill is a folder with a SKILL.md file, made of a YAML front matter (name, description, ...) and a Markdown body.
"""
import re
from dataclasses import dataclass
from pathlib import Path
import yaml

FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n(.*)\Z", re.DOTALL)  # i.e. ---\n<YAML>\n---\n<Markdown>
NAME = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")  # lowercase letters, digits, and single hyphens, not at the ends
FIELDS = {"name", "description", "license", "compatibility", "metadata", "allowed-tools"}  # the ones of the spec
RESOURCE = re.compile(r"\b(?:scripts|references|assets|evals)/[\w./-]*\w")  # e.g. scripts/check.py, in the body


def parse(path: Path) -> tuple[dict, str]:
    """The front matter (as a dict) and the body (as text) of a SKILL.md file."""
    match = FRONT_MATTER.match(path.read_text())
    if not match:
        raise ValueError(f"{path}: missing YAML front matter")
    return yaml.safe_load(match[1]) or {}, match[2].strip()


@dataclass
class Skill:
    path: Path  # the skill's folder
    front_matter: dict

    @property
    def name(self) -> str:
        return str(self.front_matter.get("name", ""))

    @property
    def description(self) -> str:
        return " ".join(str(self.front_matter.get("description", "")).split())  # on one line

    @property
    def body(self) -> str:  # read lazily: level 2 of progressive disclosure
        return parse(self.path / "SKILL.md")[1]


def discover(*folders: Path | str) -> dict[str, Skill]:
    """The skills in the given folders, by name: each sub-folder with a SKILL.md is a skill (later folders win on clashes)."""
    skills = {}
    for folder in folders:
        for file in sorted(Path(folder).glob("*/SKILL.md")):
            skill = Skill(file.parent, parse(file)[0])
            skills[skill.name] = skill
    return skills


def catalogue(skills: dict[str, Skill]) -> str:
    """What the agent sees of the skills at startup (level 1 of progressive disclosure): names and descriptions only."""
    return "\n".join(f"- {skill.name}: {skill.description}" for skill in skills.values())


def validate(skill: Skill) -> list[str]:
    """The violations of the spec (plus a few best practices) by a skill; an empty list means: valid."""
    errors = []
    if not (1 <= len(skill.name) <= 64 and NAME.fullmatch(skill.name)):
        errors.append(f"invalid name {skill.name!r}: 1-64 lowercase letters, digits, and single hyphens")
    if skill.name != skill.path.name:
        errors.append(f"the name {skill.name!r} differs from the folder's name {skill.path.name!r}")
    if not 1 <= len(skill.description) <= 1024:
        errors.append(f"the description must have 1-1024 characters, not {len(skill.description)}")
    if len(str(skill.front_matter.get("compatibility", ""))) > 500:
        errors.append("the compatibility field must have at most 500 characters")
    if unknown := set(skill.front_matter) - FIELDS:
        errors.append(f"fields not in the spec (ignored by other harnesses): {sorted(unknown)}")
    body = skill.body
    if len(body.splitlines()) > 500:
        errors.append("the body should be shorter than 500 lines: move details to references/")
    for resource in sorted(set(RESOURCE.findall(body))):  # e.g. "scripts/check.py" must exist in the skill's folder
        if not (skill.path / resource).exists():
            errors.append(f"the body mentions {resource}, which does not exist")
    return errors
