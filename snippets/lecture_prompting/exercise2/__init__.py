"""
Exercise 2: Extract Structured Information from Pictures.

TODO:
1. use an LLM with vision capabilities (e.g. Open Router's google/gemma-4-26b-a4b-it:free)
2. design a system prompt, and a user prompt with the picture of an ID document (cf. data.passport(...))
3. define a Pydantic class for the extracted information (name, date of birth, ID number, expiration date, ...)
4. call the LLM with the picture, and get the structured information
5. query the model several times for the same picture, and keep, field by field, the most voted value;
   flag fields with no clear majority for human review
6. try it with the passports of all candidates

See the slides for how to pass images to the model.

Put your solution here.
"""
