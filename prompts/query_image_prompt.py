from langchain.prompts import PromptTemplate

QUERY_IMAGE_PROMPT = """
You are an expert in visual recognition and description. 
Your task is to analyze the input cocktail image carefully and generate a detailed, clear, and concise textual description. 
Tell me the cocktail you're guessing. Include the following details whenever possible:

1. Objects and main elements in the image
2. Visual attributes: colors, shapes, textures, patterns
3. Spatial relationships and composition
4. Any notable or distinctive features

Guidelines:
- Use complete sentences and professional language.
- Be specific and avoid vague terms.
- Keep the description factual and objective.
- Focus only on what is visually present, describing it as it appears, not assumptions or interpretations.

Output the description in 3–4 sentences in English.
"""
