"""
Exercise 2: Testing Infrastructure for ID Document Extraction.

Test the extractor of Exercise 2 of the prompting lecture.

TODO:
1. first, design a test dataset: the expected values of all fields, for each of the 3 passports
2. pick a framework (DeepEval or MLflow), and write deterministic scorers
   (exact match for IDs and dates, normalised match for names; invariants, e.g. date of birth < expiration date)
3. measure consistency: run the extractor N times per picture, and report the agreement rate per field;
   does majority voting improve accuracy?
4. add an LLM-as-a-Judge only where code can't help (e.g. legibility), and validate it against your own judgement
5. run the suite with at least two models, and write down which one you'd pick, and why

Put your solution here.
"""
