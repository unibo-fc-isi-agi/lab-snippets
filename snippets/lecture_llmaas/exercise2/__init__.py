"""
Exercise 2: Retry and Exponential Backoff.

Add a configurable retry mechanism, with exponential backoff, to the Sync CLI Chat of Example 1
(e.g. copy example1/repl_chat_openai.py here, then change it).

TODO:
1. when a request fails, retry it automatically after some delay, up to a maximum number of retries,
   with delays increasing exponentially (e.g. 1s, 2s, 4s, 8s, ...)
2. make all parameters (number of retries, initial delay, backoff factor, ...) configurable,
   via env vars or command-line arguments, with smart defaults

Decision points: which failures are worth a retry? a plain loop, or a library (e.g. tenacity)?
argparse and/or os.getenv? a decorator, or a helper function?

Put your solution here.
"""
