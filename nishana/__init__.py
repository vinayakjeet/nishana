"""Nishana: a QLoRA fine-tune on one narrow Hinglish support-ticket task, scored
against a frontier baseline with the data-curation ablation that shows which part
did the work."""

from dotenv import load_dotenv

# Credentials are read from os.environ by llm/providers/base.py. Locally they
# live in .env, which nothing else exports. override=False keeps real
# environment variables winning, so CI is unchanged.
load_dotenv(override=False)
