"""Shared config constants — kept separate from agent/loop.py so other
agent/* modules (e.g. profile.py) can reference the model name without
circularly importing loop.py, which itself imports from several of them."""

MODEL = "qwen2.5-coder:7b-instruct-q4_K_M"
