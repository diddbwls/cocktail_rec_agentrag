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
The C4 algorithm searches for cocktail alternatives based on relationship and recipe complexity:
- Prioritize cocktails that share many ingredients with the target cocktail
- Filter for cocktails with similar recipe complexity (number of ingredients)
- Calculate similarity through graph relationships (HAS_INGREDIENT)

Extract the target information from the user’s question for cocktail similarity search.

## Information to Extract:
1. target_cocktail: The reference cocktail name (e.g., "Manhattan", "Mojito", "Martini")  
   - Most important! This is required for relationship-based search
2. ingredients: Explicitly mentioned ingredients (e.g., "whiskey", "vermouth", "bitters")  
   - Used to identify a target when no cocktail name is given

## Example Question 1: "recommend cocktails similar to Manhattan"
→ {{"target_cocktail": "Manhattan", "ingredients": []}}

## Example Question 2: "other cocktails in the style of Mojito"  
→ {{"target_cocktail": "Mojito", "ingredients": []}}

## Example Question 3: "cocktails with recipes similar to those using whiskey and sweet vermouth"
→ {{"target_cocktail": "", "ingredients": ["whiskey", "sweet vermouth"]}}

## Example Question 4: "cocktails similar to Old Fashioned but with different ingredients"
→ {{"target_cocktail": "Old Fashioned", "ingredients": []}}

--------------------------------
User Question: {user_question}

Reference Categories: {category_list}

Please respond strictly in the following JSON format:
{{
    "target_cocktail": "reference cocktail name (empty string if none)",
    "ingredients": ["mentioned ingredients"]
}} 
"""


# C4 System Message
C4_SYSTEM_MESSAGE = "You are a cocktail similarity expert. Always respond with valid JSON only."

C4_PROMPT_TEMPLATE = """
You are an expert in cocktail similarity and alternative recommendations.
Please answer by considering the following cocktail recommendation guidelines and answer guidelines.

## Cocktail Recommendation Guidelines
- Prioritize recommending cocktails with similar recipe complexity (number of ingredients)
- Emphasize cocktails that share the same base spirit
- Explain similarities in preparation methods (shaking vs stirring)
- Compare similarities and differences in flavor profiles
- When suggesting substitute ingredients, explain the resulting flavor changes

## Answer Guidelines:
1. Answer accurately based on the given search results.
2. Do not guess information that is not in the search results.
3. Clearly explain the similarities and differences with the target cocktail.
4. Mention ingredient overlap and recipe complexity when explaining the reason for recommendations.

Question: {question}

Search Results:
{context}

Answer:
"""