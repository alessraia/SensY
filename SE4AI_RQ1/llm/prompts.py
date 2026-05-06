INSTRUCTION = """You are a binary classifier for prompt sensitivity.

Return only one label:
0 = not sensitive
1 = sensitive
"""

def build_prompt(question: str) -> str:
    return f"""{INSTRUCTION}

Prompt:
{question}

Label:
"""