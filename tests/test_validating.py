from snippets.lecture_validating.exercise1.test_all_fields import TEST_CASES, matches, normalise
from snippets.lecture_validating.exercise2.test_id_extraction import TEST_CASES as PASSPORTS, normalise_name


def test_normalise():
    assert normalise("Maîtresse de Conférences.") == "maitresse de conferences"


def test_matches():
    assert matches("Associate Professor of Computer Science", "associate professor")
    assert matches("Université de Lyon", ["university of lyon", "universite de lyon"])
    assert matches(None, None) and matches("Not specified", None) and not matches("Italian", None)
    assert matches("Jordanian", [None, "jordan"])


def test_datasets_are_complete():
    assert {case["input"] for case in TEST_CASES} == {"letter-mario-rossi.txt", "letter-jean-dupont.txt", "letter-mohammed-ali.txt"}
    for case in TEST_CASES:
        assert set(case["expectations"]) == {"degree", "alma_mater", "attended", "affiliation", "position", "seniority", "nationality", "application_for"}
    assert all(set(case["expected"]) == {"name", "nationality", "date_of_birth", "id_number", "expiration_date", "legible"} for case in PASSPORTS)


def test_normalise_name():
    assert normalise_name("DUPONT, Jean") == normalise_name("Jean Dupont")
