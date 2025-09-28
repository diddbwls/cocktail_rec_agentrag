# C2 태스크: Glass Type + 재료 매칭 기반 검색

from langchain.prompts import PromptTemplate
# C2 키워드 추출 프롬프트 - Glass Type + 재료
C2_KEYWORD_EXTRACTION_PROMPT = """
Extract keywords for glass-type–first cocktail search.

Guidelines:
- cocktail: ONLY explicitly named cocktails (no inference).
- include_ingredients: explicitly required ingredients.
- exclude_ingredients: explicitly excluded ingredients (e.g., without/ exclude).
- glassType: explicit glass types (e.g., "martini glass", "highball glass", "rocks glass").
- category: pick ONLY from DB list: Coffee / Tea, Cocoa, Milk / Float / Shake, Soft Drink, Shot, Cocktail, Shake,
            Other / Unknown, Punch / Party Drink, Homemade Liqueur, Ordinary Drink, Beer.

Examples:
"I want a cocktail similar to Martini served in a martini glass"
→ {{"cocktail":["Martini"],"include_ingredients":[],"exclude_ingredients":[],"glassType":["martini glass"],"category":[]}}
"cocktail with gin and lime in a highball glass"
→ {{"cocktail":[],"include_ingredients":["gin","lime"],"exclude_ingredients":[],"glassType":["highball glass"],"category":[]}}
"Cocktail category in a rocks glass made without whiskey"
→ {{"cocktail":[],"include_ingredients":[],"exclude_ingredients":["whiskey"],"glassType":["rocks glass"],"category":["Cocktail"]}}

User Question: "{question}"
Categories: {category_list}

Output VALID JSON only:
{{
  "cocktail": [],
  "include_ingredients": [],
  "exclude_ingredients": [],
  "glassType": [],
  "category": []
}}
"""


# System message
C2_SYSTEM_MESSAGE = (
  "You are a keyword extraction expert for cocktail search. "
  "Extract ONLY explicitly mentioned cocktail names, ingredients, glass types, and categories. "
  "Never infer. Output valid JSON only."
)

C2_PROMPT_TEMPLATE = """
CRITICAL: Use ONLY the search results below. Do NOT add outside info.

Task: Recommend cocktails prioritized by glass type, then match ingredients.

Explain briefly:
- How the specified glass type relates to serving style and typical cocktails.
- Commonalities/differences among cocktails using that glass.
- Level-based ingredient matching results (all-match → partial-match) and why they fit.

Answer Rules:
- Strictly stick to the provided context; no guessing.
- Be concise: summarize in 3–5 sentences.

Question: {question}

Search Results:
{context}

Answer:
"""
