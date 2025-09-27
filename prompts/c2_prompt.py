# C2 태스크: Glass Type + 재료 매칭 기반 검색

from langchain.prompts import PromptTemplate
# C2 키워드 추출 프롬프트 - Glass Type + 재료
C2_KEYWORD_EXTRACTION_PROMPT = """
The C2 algorithm searches for cocktails based on glass type priority and ingredient matching.
Please classify the keywords according to the following guidelines, classification rules, and examples.

## Guidelines:
- cocktail: Only include cocktail names that are explicitly mentioned (e.g., "Martini", "Negroni", "Mojito"). Do not infer.
- include_ingredients: Ingredients that must be included (e.g., "gin", "whiskey", "lime")
- exclude_ingredients: Ingredients that must be excluded (when expressed with "without", "exclude", "leave out", etc.)
- glassType: Glass types (e.g., "martini glass", "highball glass", "rocks glass")
- category: Must select only from the following DB category list:
  * Coffee / Tea
  * Cocoa
  * Milk / Float / Shake
  * Soft Drink
  * Shot
  * Cocktail
  * Shake
  * Other / Unknown
  * Punch / Party Drink
  * Homemade Liqueur
  * Ordinary Drink
  * Beer

## Example Question 1: "I want a cocktail similar to Martini served in a martini glass"
→ {{"cocktail": ["Martini"], "include_ingredients": [], "exclude_ingredients": [], "glassType": ["martini glass"], "category": []}}

## Example Question 2: "cocktail with gin and lime served in a highball glass"  
→ {{"cocktail": [], "include_ingredients": ["gin", "lime"], "exclude_ingredients": [], "glassType": ["highball glass"], "category": []}}

## Example Question 3: "Cocktail category in a rocks glass made without whiskey"
→ {{"cocktail": [], "include_ingredients": [], "exclude_ingredients": ["whiskey"], "glassType": ["rocks glass"], "category": ["Cocktail"]}}

## Example Question 4: "cocktail with vodka and cranberry in a cocktail glass"
→ {{"cocktail": [], "include_ingredients": ["vodka", "cranberry"], "exclude_ingredients": [], "glassType": ["cocktail glass"], "category": []}}

## Example Question 5: "cocktail in Negroni style with orange included"
→ {{"cocktail": ["Negroni"], "include_ingredients": ["orange"], "exclude_ingredients": [], "glassType": [], "category": []}}

--------------------------------
User Question: "{question}"

Categories: {category_list}

Please classify strictly in the following JSON format:
{{
  "cocktail": [],
  "include_ingredients": [],
  "exclude_ingredients": [],
  "glassType": [],
  "category": []
}}

"""


# System message
C2_SYSTEM_MESSAGE = "You are a keyword extraction expert for cocktail search. ONLY extract directly mentioned cocktail names, ingredients, glass types, and categories. Never infer or guess cocktail names from descriptions, colors, or characteristics. Always respond with valid JSON only."

C2_PROMPT_TEMPLATE = """
**IMPORTANT: You must ONLY use information provided in the search results. Do NOT add any information not present in the context.**
You are an expert in matching glass types and ingredients.
Please answer by considering the following cocktail recommendation guidelines and answer guidelines.

## Cocktail Recommendation Guidelines
- Describe cocktail characteristics and serving styles by glass type
- Commonalities and differences among cocktails that use the same glass
- Level-based ingredient matching (progressively expand from all ingredients → partial ingredients)
- Glass usage patterns by cocktail family

## Answer Guidelines:
1. Answer accurately based on the given search results.
2. Do not guess information that is not in the search results.
3. Focus on the relationship between glass type and ingredient combinations.
4. Systematically organize the results of level-based ingredient matching.
5. **IMPORTANT: You must ONLY use information provided in the search results. Do NOT add any information not present in the context.**

Question: {question}

Search Results:
{context}

Answer:
"""

