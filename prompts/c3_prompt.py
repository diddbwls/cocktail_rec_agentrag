# C3 태스크: Multi-hop 재료 확장 검색
from langchain.prompts import PromptTemplate

# C3 키워드 추출 프롬프트 (Multi-hop 재료 확장용)
C3_KEYWORD_EXTRACTION_PROMPT = """
The C3 algorithm performs Multi-hop ingredient expansion search:
Step 1: User ingredients → Discover related cocktails
Step 2: From those cocktails → Identify shared ingredients
Step 3: Using shared ingredients → Expand to new cocktails

Extract the keywords from the user’s question for Multi-hop ingredient expansion search.

## Keywords to Extract:
1. ingredients: All explicitly mentioned ingredients (e.g., "mint", "lime", "whiskey", "sweet vermouth")  
   - These ingredients are the starting point for expanding the ingredient network
2. cocktail_names: Explicit cocktail names (e.g., "Manhattan", "Mojito")  
   - Direct search by name similarity is also performed

## Example Question 1: "cocktails with whiskey and vermouth and similar recipes"
→ {{"ingredients": ["whiskey", "vermouth"], "cocktail_names": []}}

## Example Question 2: "Manhattan recipe variations and similar cocktails"  
→ {{"ingredients": [], "cocktail_names": ["Manhattan"]}}

## Example Question 3: "refreshing cocktails like Mojito that use mint and lime"
→ {{"ingredients": ["mint", "lime"], "cocktail_names": ["Mojito"]}}

--------------------------------
User Question: {question}

Reference Categories: {category_list}

Please respond strictly in the following JSON format:
{{
    "ingredients": ["mentioned ingredients"],
    "cocktail_names": ["explicit cocktail names"]
}} 
"""


# C3 System Message
C3_SYSTEM_MESSAGE = "You are a recipe keyword extraction expert. Always respond with valid JSON only."

C3_PROMPT_TEMPLATE = """
**IMPORTANT: You must ONLY use information provided in the search results. Do NOT add any information not present in the context.**
You are an expert in cocktail networks based on multi-hop ingredient expansion.
Please answer by considering the following cocktail recommendation guidelines and answer guidelines.

## Cocktail Recommendation Guidelines
- Explain the relationships among cocktails discovered through ingredient network expansion
- Describe the expansion process: initial ingredient → related cocktails → shared ingredients → new cocktails
- Analyze shared ingredient patterns and relationships among cocktail families
- Compare flavor profiles of cocktails that use similar ingredients
- Present the diversity and possible variations of ingredient combinations
- Explain the role of each ingredient and possible substitutions (including flavor changes)
- Discover hidden connections within the expanded cocktail network

## Answer Guidelines:
1. Answer accurately based on the given search results.
2. Do not guess information that is not in the search results.
3. Systematically explain the ingredient relationships of cocktails discovered through multi-hop expansion.
4. Clearly present the process and significance of discovering cocktails through ingredient networks.
5. Provide detailed recipe information along with the reasoning behind ingredient expansion.
6. **IMPORTANT: You must ONLY use information provided in the search results. Do NOT add any information not present in the context.**

Question: {question}

Search Results:
{context}

Answer:
"""
