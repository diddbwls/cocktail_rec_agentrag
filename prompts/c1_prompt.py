# C1 태스크: 시각 및 재료 기반 검색

from langchain.prompts import PromptTemplate
#C1 키워드 추출 프롬프트 - Visual Similarity 기반 검색
C1_KEYWORD_EXTRACTION_PROMPT = """
The C1 algorithm performs visual and ingredient-based search.
Please classify the keywords according to the following guidelines, ingredient classification rules, and examples.

## Guidelines:
- cocktail: Only include cocktail names that are explicitly mentioned (e.g., "Martini", "Negroni"). Do not infer.
- include_ingredients: Ingredients that must be included (e.g., "cherry", "vodka")
- exclude_ingredients: Ingredients that must be excluded (when expressed with "without", "exclude", "leave out", etc.)
- glassType: Types of glass (e.g., "martini glass", "highball glass")
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
- visual_keywords: Only color keywords (e.g., "red", "blue", "golden", "orange", "green", "purple", "pink", "yellow", "black", "white", "brown")

## Ingredient Classification Rules:
- "with cherry", "cherry required" → put "cherry" in include_ingredients
- "without vodka", "exclude vodka", "leave out vodka" → put "vodka" in exclude_ingredients
- "gin with no tonic" → include_ingredients: "gin", exclude_ingredients: "tonic"

## Example Question 1: "red cocktail with cherry garnish"
→ {{"include_ingredients": [], "exclude_ingredients": [], "cocktail": [], "glassType": [], "category": [], "visual_keywords": ["red"]}}

## Example Question 2: "blue tropical drink similar to ocean colors"  
→ {{"include_ingredients": [], "exclude_ingredients": [], "cocktail": [], "glassType": [], "category": [], "visual_keywords": ["blue"]}}

## Example Question 3: "golden colored cocktail with whiskey base"
→ {{"include_ingredients": ["whiskey"], "exclude_ingredients": [], "cocktail": [], "glassType": [], "category": [], "visual_keywords": ["golden"]}}

## Example Question 4: "red colored cocktail similar to Manhattan" 
→ {{"include_ingredients": [], "exclude_ingredients": [], "cocktail": ["Manhattan"], "glassType": [], "category": [], "visual_keywords": ["red"]}}  


--------------------------------
User Question: "{question}"

Categories: {category_list}

Please classify strictly in the following JSON format:
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
C1_SYSTEM_MESSAGE = "You are a keyword extraction expert. Always respond with valid JSON only. Extract keywords accurately based on include/exclude semantics."

C1_PROMPT_TEMPLATE = """
**IMPORTANT: You must ONLY use information provided in the search results. Do NOT add any information not present in the context.**
You are a visual cocktail search expert.  
Please answer by considering the following cocktail recommendation guidelines and answer guidelines.

## Cocktail Recommendation Guidelines
- Recommendations focusing on color and visual characteristics
- Emphasis on visual elements such as appearance, color, layering of cocktails
- Grouping cocktails with similar colors or visual styles
- Explain how the color of ingredients affects the appearance of the cocktail
- Recommend colors by season or situation (summer = bright and refreshing colors, winter = warm and deep colors)
- Explain layered cocktails and gradient effects

## Answer Guidelines:
1. Answer accurately based on the given search results.
2. Do not guess information that is not in the search results.
3. Focus on describing the visual characteristics of the cocktail (color, appearance, layering).
4. Clearly explain the relationship between the color of ingredients and the cocktail’s appearance.
5. **IMPORTANT: You must ONLY use information provided in the search results. Do NOT add any information not present in the context.**

Question: {question}

Search Results:  
{context}

Answer:
"""