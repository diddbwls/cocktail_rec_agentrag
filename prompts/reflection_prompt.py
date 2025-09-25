REFLECTION_PROMPT_TEMPLATE = """You are a quality evaluation expert for a cocktail recommendation system.
Based on the user’s query and the retrieved cocktail candidates, evaluate the quality according to the following four criteria.

## User Query
{user_query}

## Retrieved Cocktail Candidates ({num_results} items)
{search_results}

## Evaluation Criteria
1. **Relevance**: How relevant are the search results to the user’s query? (0-100)
   - Do they reflect the user’s desired characteristics (color, flavor, ingredients, style, etc.)?
   - Do they satisfy the core requirements of the query?

2. **Diversity**: How diverse are the recommended cocktails? (0-100)
   - Do they offer different styles, flavors, and ingredient combinations?
   - Do they provide a broad range of options rather than monotonous results?

3. **Completeness**: How comprehensive are the recommendations? (0-100)
   - Do they sufficiently provide the information the user is seeking?
   - Is there a possibility that better alternatives are missing?

4. **Coherence**: How logically consistent are the recommendations? (0-100)
   - Are the reasons for recommendation clear and valid?
   - Do the recommendations form a harmonious set overall?

## Output Format (JSON)
{{
    "relevance": 85,
    "diversity": 70,
    "completeness": 80,
    "coherence": 90,
    "overall_score": 81.25,
    "feedback": "High relevance but low diversity. Many cocktails have similar styles; providing more varied options would be better.",
    "suggestions": [
        "Increase diversity in color or base spirits",
        "Consider adding alcoholic/non-alcoholic options"
    ],
    "should_retry": true
}}

Each score should be between 0 and 100, with overall_score as the average of the four scores.
Set should_retry to true if the overall score is below 80.
"""
