# C3 태스크: Multi-hop 재료 확장 검색
from langchain.prompts import PromptTemplate

# C3 키워드 추출 프롬프트 (Multi-hop 재료 확장용)
C3_KEYWORD_EXTRACTION_PROMPT = """
Extract keywords for multi-hop ingredient expansion.

Process:
1) From user ingredients → related cocktails
2) From those cocktails → shared ingredients
3) Using shared ingredients → new cocktails

Extract ONLY explicitly mentioned terms.
- ingredients: all explicit ingredients
- cocktail_names: explicit cocktail names
Output VALID JSON only.

Examples:
"cocktails with whiskey and vermouth"
→ {{"ingredients":["whiskey","vermouth"],"cocktail_names":[]}}
"Manhattan variations"
→ {{"ingredients":[],"cocktail_names":["Manhattan"]}}
"like Mojito that use mint and lime"
→ {{"ingredients":["mint","lime"],"cocktail_names":["Mojito"]}}

User Question: {question}
Reference Categories: {category_list}

JSON:
{{
  "ingredients": ["mentioned ingredients"],
  "cocktail_names": ["explicit cocktail names"]
}}
"""


# C3 System Message
C3_SYSTEM_MESSAGE = (
  "You are a keyword extraction expert for ingredient-based cocktail search. "
  "Never infer. Always output valid JSON only."
)

C3_PROMPT_TEMPLATE = """
CRITICAL: Use ONLY the search results below. Do NOT add outside info.
CRITICAL: Only describe what is explicitly provided in the search results.
CRITICAL: Do not make claims about flavors, taste differences, or substitution effects unless explicitly stated.

Task: Explain multi-hop ingredient expansion results and recommend relevant cocktails.

Explain briefly:
- The ingredient expansion chain shown in the search results.
- Shared ingredient patterns among the recommended cocktails.
- How the expanded cocktails relate to the user's original request through common ingredients.

Answer Rules:
- Strictly follow the provided context; no guessing or assumptions.
- Do not invent flavor profiles, taste descriptions, or substitution effects.
- Be concise: summarize in 3–5 sentences.

Question: {question}

Search Results:
{context}

Answer:
"""