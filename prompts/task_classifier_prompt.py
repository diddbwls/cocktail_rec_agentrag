from langchain.prompts import PromptTemplate

TASK_CLASSIFIER_TEMPLATE = """
You are an expert in classifying cocktail-related questions. Analyze the user’s question and classify it into exactly one of the following four tasks.

**Task Definitions:**

**C1: Color–Ingredient Visual Search**
- Questions containing color keywords ("red", "blue", "golden", "green", etc.)
- Visual appearance, similar/visually alike cocktails
- Visual adjectives such as "elegant appearance", "beautiful", "visually appealing"
- Questions about the relationship between color and ingredients
- Questions related to color contrast or visual harmony

**C2: Glass Type + Ingredient Matching**
- Questions explicitly mentioning a specific glass type ("highball glass", "martini glass", "rocks glass", etc.)
- Relationship between multiple ingredient combinations and glass types
- Cocktail recommendations by glass type
- Glass-related keywords such as "served in", "glass", "tumbler"
- Composite conditions of cocktail + glass + ingredients

**C3: Multi-hop Ingredient Expansion Search**
- Questions that provide a **list of ingredients** (without specifying a cocktail name)
- "Cocktails made with whiskey and vermouth"
- Ingredient → cocktail → shared ingredient → new cocktail exploration
- "What can I make with these ingredients?"
- Discovery-based search based on ingredient combinations

**C4: Cocktail Recipe Similarity and Alternative Recommendation**
- Questions explicitly mentioning a **target cocktail name**
- "Cocktails with recipes similar to Manhattan", "A recipe like Mojito"
- "Alternatives to Old Fashioned"
- Alternatives that share many ingredients with the target cocktail
- Recommendations for cocktails with similar recipe complexity (number of ingredients)

**Classification Examples:**

**C1 Examples:**
- "red layered cocktail with elegant appearance" → C1 (color + visual feature)
- "golden colored cocktail with whiskey base" → C1 (color keyword focused)
- "blue tropical drink similar to ocean colors" → C1 (color-based visual search)

**C2 Examples:**
- "cocktail with gin and lime served in a highball glass" → C2 (glass + ingredient combo)
- "I want a cocktail similar to Martini but served in a martini glass" → C2 (glass-centered)
- "cocktail with vodka and cranberry in a cocktail glass" → C2 (glass + ingredient)

**C3 Examples:**
- "cocktails with whiskey and vermouth" → C3 (ingredient list only)
- "find cocktails that use mint and lime" → C3 (ingredient combo → cocktail discovery)
- "what can I make with vodka and cranberry?" → C3 (ingredient → cocktail exploration)

**C4 Examples:**
- "recommend cocktails with recipes similar to Manhattan" → C4 (target = Manhattan specified)
- "other cocktails with a recipe like Mojito" → C4 (target = Mojito)
- "cocktail recipes to drink instead of Old Fashioned" → C4 (target = Old Fashioned alternative)

**Answer Format:**
You must answer strictly in the following JSON format:
{{
    "task": "C1",
    "confidence": 85,
    "reason": "The question is asking for visual similarity and recommendation, so C1 is appropriate."
}}

Question: {question}

Classification:
"""


# 태스크 분류를 위한 프롬프트
task_classifier_prompt = PromptTemplate(
    template=TASK_CLASSIFIER_TEMPLATE,
    input_variables=["question"]
)