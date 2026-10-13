"""
A CLI chat with an LLM, via OpenAI's Responses API, where files (pictures, PDFs, text) can be attached to messages.
Answers are streamed. By default, the history is kept by us (as with Chat Completions): with --stateful, by the provider.

Run with: poetry run python -m snippets -l llmaas -e 3 [--stateful]
then, e.g.: /attach data/passport-mario-rossi.png, and ask: "What is the nationality of this person?"
Configure via env vars: OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL (must support images, for pictures).
Only some providers support the Responses API (e.g. OpenAI, Open Router, Ollama), and only OpenAI supports --stateful
(others may silently ignore it, e.g. Ollama: then the model forgets everything at each message).
"""
import argparse
import base64
import mimetypes
import os
from pathlib import Path
from openai import OpenAI

parser = argparse.ArgumentParser(description="CLI Chat with Attachments, via the Responses API")
parser.add_argument("--stateful", action="store_true", help="let the provider keep the history (OpenAI only)")
args = parser.parse_args()

base_url = os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1/")
api_key = os.environ.get("OPENAI_API_KEY") or input(f"Enter your API key for {base_url}: ")
model = os.environ.get("OPENAI_MODEL", "google/gemma-4-26b-a4b-it:free")  # a free model that can see pictures

client = OpenAI(base_url=base_url, api_key=api_key)
INSTRUCTIONS = "Just chat with the user, be friendly and helpful. Answer questions about the attached files, if any."


def attachment(path: Path) -> dict:
    """Turns a file into a content part of a message: pictures and PDFs are sent in base64, text files inline."""
    mime = mimetypes.guess_type(path)[0] or "text/plain"
    if mime.startswith("text/"):
        return dict(type="input_text", text=f"<file name='{path.name}'>\n{path.read_text()}\n</file>")
    data_url = f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"
    if mime.startswith("image/"):
        return dict(type="input_image", image_url=data_url)
    return dict(type="input_file", filename=path.name, file_data=data_url)  # e.g. PDFs


history = []          # stateless mode: all messages so far, sent at each request
previous_id = None    # stateful mode: the ID of the last response, stored by the provider
attachments = []      # files to be sent along with the next message

print(f"Using model: {model} ({'stateful' if args.stateful else 'stateless'})")
print("Type '/exit' or '/quit' to stop. Type '/attach <path>' to attach a file to your next message.")

try:
    while True:
        user_text = input("you> ").strip()
        if not user_text:
            continue
        cmd, _, arg = user_text.partition(" ")
        if cmd.lower() in {"/exit", "/quit"}:
            break
        if cmd.lower() == "/attach":
            if not (path := Path(arg.strip())).is_file():
                print(f"error> no such file: {path}")
                continue
            attachments.append(attachment(path))
            print(f"(attached {path.name}: it will be sent along with your next message)")
            continue
        # a message is a list of content parts: the attachments, then the text
        message = dict(role="user", content=[*attachments, dict(type="input_text", text=user_text)])
        if args.stateful:  # we only send the new message, and point to the previous response (the provider has the rest)
            request = dict(input=[message], previous_response_id=previous_id, store=True)
        else:  # we send the whole history, and ask the provider NOT to store anything
            request = dict(input=[*history, message], store=False)
        answer = ""
        try:
            stream = client.responses.create(model=model, instructions=INSTRUCTIONS, stream=True, **request)
            print("assistant> ", end="", flush=True)
            for event in stream:  # typed events: we only need two kinds of them
                if event.type == "response.created":
                    response_id = event.response.id
                elif event.type == "response.output_text.delta":
                    print(event.delta, end="", flush=True)
                    answer += event.delta
            print()
        except Exception as exc:  # e.g. network errors, rate limits, unsupported features
            print(f"\nerror> {exc}")
            continue
        attachments = []  # sent: don't send them again
        previous_id = response_id
        history += [message, dict(role="assistant", content=answer)]
except (EOFError, KeyboardInterrupt):
    pass
print("Goodbye!")
