"""
Exercise 2: Tools to Take Decisions.

TODO:
1. start from Exercise 1, and add write-enabled tools: record_decision(candidate, decision, motivation),
   schedule_interview(candidate, when), send_email(candidate, subject, body) (simulated, into an outbox/ directory)
2. make sure no write-enabled tool is executed without approval (e.g. LangChain's human-in-the-loop middleware):
   the human sees the full call, and may approve, edit, or reject it
3. write a malicious letter for a fourth candidate, containing an injected instruction
4. extend the trajectory tests: no write without approval, no write upon mere questions, no action due to the malicious letter

Put your solution here.
"""
