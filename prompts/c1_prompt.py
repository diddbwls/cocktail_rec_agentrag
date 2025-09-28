# C1 태스크: 시각 및 재료 기반 검색

from langchain.prompts import PromptTemplate
#C1 키워드 추출 프롬프트 - Visual Similarity 기반 검색
C1_KEYWORD_EXTRACTION_PROMPT = """
Extract keywords for visual + ingredient-based cocktail search.

Guidelines:
- cocktail: ONLY explicitly named cocktails (no inference).
- include_ingredients: explicitly required ingredients.
- exclude_ingredients: explicitly excluded ingredients ("without", "exclude", "leave out").
- glassType: explicit glass types (e.g., martini glass, highball glass).
- category: pick ONLY from DB list: Coffee / Tea, Cocoa, Milk / Float / Shake, Soft Drink,
            Shot, Cocktail, Shake, Other / Unknown, Punch / Party Drink, Homemade Liqueur,
            Ordinary Drink, Beer.
- visual_keywords: ONLY explicit color terms (red, blue, golden, orange, green,
                   purple, pink, yellow, black, white, brown).

Examples:
"red cocktail with cherry garnish"
→ {{"cocktail":[],"include_ingredients":["cherry"],"exclude_ingredients":[],"glassType":[],"category":[],"visual_keywords":["red"]}}
"blue tropical drink"
→ {{"cocktail":[],"include_ingredients":[],"exclude_ingredients":[],"glassType":[],"category":[],"visual_keywords":["blue"]}}
"golden cocktail with whiskey base"
→ {{"cocktail":[],"include_ingredients":["whiskey"],"exclude_ingredients":[],"glassType":[],"category":[],"visual_keywords":["golden"]}}
"red cocktail similar to Manhattan"
→ {{"cocktail":["Manhattan"],"include_ingredients":[],"exclude_ingredients":[],"glassType":[],"category":[],"visual_keywords":["red"]}}

User Question: "{question}"
Categories: {category_list}

Output VALID JSON only:
{{
  "cocktail": [],
  "include_ingredients": [],
  "exclude_ingredients": [],
  "glassType": [],
  "category": [],
  "visual_keywords": []
}}
"""

# 시스템 메시지
C1_SYSTEM_MESSAGE = (
  "You are a keyword extraction expert for visual cocktail search. "
  "Extract ONLY explicitly mentioned cocktail names, ingredients, glass types, categories, and colors. "
  "Never infer. Always output valid JSON only."
)

C1_PROMPT_TEMPLATE = """
CRITICAL: Use ONLY the search results below. Do NOT add outside info.
CRITICAL: Only describe what is explicitly provided in the search results.
CRITICAL: Do not make claims about colors, appearance, or visual properties unless explicitly stated.

Task: Recommend cocktails based on ingredient similarity and glass types.

Explain briefly:
- Shared ingredients among the recommended cocktails.
- How the ingredients and glass types relate to the user's request.
- Commonalities among cocktails with similar ingredient profiles.

Answer Rules:
- Stick strictly to the context; no guessing or assumptions.
- Do not invent color, flavor, or visual descriptions.
- Be concise: summarize in 3–5 sentences.

Question: {question}

Search Results:
{context}

Answer:
"""