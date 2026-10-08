from datetime import date
from snippets.lecture_prompting.exercise1.letter_scoring_checklist import LetterInfo
from snippets.lecture_prompting.exercise2.id_extraction import IDDocumentInfo, flagged, vote


def letter(**checklist) -> LetterInfo:
    criteria = dict(clear_acquaintance=True, storytelling=True, concrete_skills=True, concrete_strengths=True, no_weaknesses=True, tailored=True)
    return LetterInfo(
        applicant=dict(name="A", degree=None, alma_mater=None, attended=[], skills=["x"], strengths=["y"], weaknesses=[]),
        author=dict(name="B", affiliation="U", position=None, seniority=None, email="b@u", nationality=None, relationship_with_applicant=None),
        application_for="AI",
        checklist=criteria | checklist,
    )


def test_score_is_computed_from_checklist():
    assert letter().score == 5 and letter().penalties == []
    info = letter(tailored=False, no_weaknesses=False)
    assert info.score == 3 and set(info.penalties) == {"tailored", "no_weaknesses"}


def test_missing_fields_are_penalised():
    info = letter()
    info.author.email = ""
    assert info.score == 4 and info.penalties == ["missing author.email"]


def test_score_is_never_negative():
    assert letter(**dict.fromkeys(["clear_acquaintance", "storytelling", "concrete_skills", "concrete_strengths", "no_weaknesses", "tailored"], False)).score == 0


def passport(name="Mario Rossi", id_number="BN1234567") -> IDDocumentInfo:
    return IDDocumentInfo(name=name, nationality="Bananense", date_of_birth=date(1985, 3, 15), id_number=id_number, expiration_date=date(2034, 5, 20))


def test_vote_field_by_field():
    info, agreement = vote([passport(), passport(name="Mario Rosi"), passport(id_number="BN1234561"), passport(id_number="BN1234562")])
    assert info.name == "Mario Rossi" and agreement["name"] == 0.75
    assert agreement["id_number"] == 0.5
    assert flagged(agreement) == ["id_number"]  # no absolute majority
