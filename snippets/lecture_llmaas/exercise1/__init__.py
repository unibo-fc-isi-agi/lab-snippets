"""
Exercise 1: Caching Sync Requests.

Add a file-system cache of Chat Completion requests to the Sync CLI Chat of Example 1
(e.g. copy example1/repl_chat_openai.py here, then change it).

TODO:
1. cache Chat Completion requests on the file system, before issuing them
2. upon a new request, check whether the same request was already made: if so, return the cached response,
   instead of calling the model (and print something like "Cache hit!")

Decision points: where to store the cache? when is it a hit (same last message? same history? same model and parameters?)
how to store it (e.g. JSON files, named after the cache index)? how to restructure the code?

Put your solution here.
"""
