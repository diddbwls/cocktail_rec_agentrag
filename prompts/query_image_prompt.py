from langchain.prompts import PromptTemplate
QUERY_IMAGE_PROMPT = """
CRITICAL: Extract only 1–2 short, simple English sentences focusing on the key visual elements of the cocktail image.

Include only:
- Main objects (glass, garnish, liquid, etc.)
- Visual attributes (color, shape, texture)
- Distinctive or notable features

Guidelines:
- Be strictly factual and specific, not interpretive.
- Describe only what is visually present, nothing more.
"""
