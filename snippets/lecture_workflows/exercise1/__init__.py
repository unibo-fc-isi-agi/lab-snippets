"""
Exercise 1: a Bot for Candidates' E-mails.

A semi-automatic assistant of the PhD admission office, answering the candidates' e-mails as a LangGraph workflow.

TODO:
1. model the bot as a LangGraph workflow, with (at least):
   - a routing step (LLM, structured output): questions on the regulations, on the status of the sender's application,
     out of scope, suspicious
   - per route: RAG over the regulations (cf. search_regulations in snippets/lecture_workflows/subagents.py),
     a data step reading the application's record (data.applications()), a canned reply, an escalation to a human
   - a drafting step, and an evaluator-optimizer loop (no invented facts, no data of other candidates, cite the articles)
   - a human-in-the-loop step (approve / edit / reject), before an idempotent send_reply (writing to an outbox/)
2. persist runs with a SQLite checkpointer: one thread per e-mail (data.inbox()), approvals may come later
3. stream progress to the CLI, and trace runs (e.g. with MLflow)
4. write tests at the three levels (step, path, end-to-end)

Put your solution here.
"""
