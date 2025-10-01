"""
Shared prompt template for LLM-as-a-Judge evaluation of cocktail recommendations.
"""

def get_evaluation_prompt(answer: str, query: str, context: str = "") -> str:
    """
    Generate evaluation prompt for cocktail recommendation explanations.
    
    Args:
        answer: The explanation/recommendation text to evaluate
        query: The user's original query
        context: Additional context information (optional)
    
    Returns:
        Formatted prompt string for LLM evaluation
    """
    
    prompt = f"""
Considering you are a user of a cocktail recommendation platform.
The recommendation system has received your question and suggested a cocktail, accompanied by an explanation.
Please rate the user experience with the explanation in the following aspects:

- Persuasiveness: This explanation is convincing to me.
- Transparency: Based on this explanation, I understand why this cocktail is recommended.
- Accuracy: This explanation is consistent with my taste/preferences.
- Satisfaction: I am satisfied with this explanation.

Assess the aspects with integers between 1–5, where 1 = strongly disagree and 5 = strongly agree.

Query: {query}
Answer: {answer}

Return ONLY a compact JSON object with these integer fields (1–5, no other text, no trailing comments):
{{"persuasiveness": int, "transparency": int, "accuracy": int, "satisfaction": int}}

"""
    
    return prompt

#Context: {context}