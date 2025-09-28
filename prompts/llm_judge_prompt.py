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
You are a customer using a cocktail recommendation platform.
The system has suggested a cocktail to you, along with an explanation text.
Your task is to evaluate the quality of this explanation from the perspective of an end user.

Judge the explanation strictly based on the provided query and context.
Do not use any knowledge outside the provided information.
Return JSON only. Do not add explanations or extra text.

Criteria (1–5 Likert scale):
1. Persuasiveness: "This explanation is convincing to me."
2. Transparency: "Based on this explanation, I understand why this cocktail is recommended."
3. Accuracy: "This explanation is consistent with the provided query and context." 
   - If no context is given, evaluate consistency only with the query.
4. Satisfaction: "I am satisfied with this explanation."

Rating Scale:
1 = Strongly disagree, 2 = Disagree, 3 = Neutral, 4 = Agree, 5 = Strongly agree

Return your evaluation strictly in JSON format:
{{"persuasiveness": 1, "transparency": 1, "accuracy": 1, "satisfaction": 1}}

Query: {query}
Context: {context}
Answer: {answer}

"""
    
    return prompt