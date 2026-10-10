"""
Exercise 2: the E-mail Bot as an A2A Agent.

The bot of Exercise 1, exposed via A2A, so that other agents (e.g. a university's help desk) can delegate candidates'
e-mails to it, and follow each one as a task, until the admission office approves (or rejects) the reply.

TODO:
1. write the bot's Agent Card (cf. Example 4 bis): name, description, skills, input/output modes
2. write an A2A server (with a2a-sdk), whose executor runs the workflow of Exercise 1: one A2A task per e-mail,
   one LangGraph thread per task
3. map the workflow onto the task lifecycle: working, then waiting for the office's approval (working, or input-required?),
   finally completed (with the reply as an artifact), or rejected / failed
4. write a small A2A client (e.g. the help desk), which sends an e-mail and streams the task's status updates
5. test it: an approved e-mail ends completed with the reply; a rejected one does not; the injection e-mail never leaks data

Put your solution here.
"""
