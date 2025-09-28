# C4 태스크: 재료기반 유사 레시피 칵테일 추천
# c4 예시 질문들:
# 1. "Manhattan과 유사한 칵테일 추천해줘"
# 2. "Mojito 같은 스타일의 다른 칵테일들"
# 3. "위스키와 베르무스를 사용하는 칵테일과 비슷한 레시피"
# 4. "복잡하지 않은 간단한 칵테일로 Martini 대안"
# 5. "Old Fashioned와 비슷하지만 다른 재료를 사용하는 칵테일"

from langchain.prompts import PromptTemplate

# C4 키워드 추출 프롬프트
C4_KEYWORD_EXTRACTION_PROMPT = """
Extract target info for cocktail recipe-similarity search.

Rules:
- target_cocktail: name if present; else "".
- ingredients: explicitly mentioned only.
- Output VALID JSON only.

Examples:
Q: "recommend cocktails similar to Manhattan"
→ {{"target_cocktail":"Manhattan","ingredients":[]}}
Q: "cocktails using whiskey and sweet vermouth"
→ {{"target_cocktail":"","ingredients":["whiskey","sweet vermouth"]}}

User Question: {question}
Reference Categories: {category_list}

JSON:
{{
  "target_cocktail": "name or empty string",
  "ingredients": ["mentioned ingredients"]
}} 
"""

C4_SYSTEM_MESSAGE = (
  "You are a cocktail recipe recommendation expert."
  "Never infer. Always output valid JSON only."
)



C4_PROMPT_TEMPLATE = """
CRITICAL: Use ONLY the search results below. Do NOT add outside info.
CRITICAL: Only describe what is explicitly provided in the search results.
CRITICAL: Do not make claims about preparation methods, flavor profiles, or characteristics unless explicitly stated.

Task: Recommend cocktails with similar recipe patterns based on the search results.

Explain briefly:
- Ingredient count and complexity similarities among the recommended cocktails.
- Shared base spirits and common ingredients in the search results.
- How the recommended cocktails relate to the user's request through ingredient patterns.

Answer Rules:
- Base everything strictly on the search results.
- Do not guess preparation methods (shaking vs stirring) or flavor descriptions.
- Do not invent substitution effects or taste comparisons.
- Be concise: summarize in 3–5 sentences.

Question: {question}

Search Results:
{context}

Answer:
"""