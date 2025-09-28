from langchain.prompts import PromptTemplate
QUERY_IMAGE_PROMPT = """
Describe the cocktail image in 1–2 simple English sentences, focusing only on key visual elements.
Include:
- Main objects (glass, garnish, liquid, etc.)
- Visual attributes (color, shape, texture)
- Distinctive or notable features

Guidelines:
- Be factual and specific, not interpretive.
- Describe only what is visually present.
"""
