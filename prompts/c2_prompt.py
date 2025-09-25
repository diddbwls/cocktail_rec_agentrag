# C2 태스크: Glass Type + 재료 매칭 기반 검색

from langchain.prompts import PromptTemplate
# C2 키워드 추출 프롬프트 - Glass Type + 재료
C2_KEYWORD_EXTRACTION_PROMPT = """
C2 알고리즘은 Glass Type 우선순위와 재료 매칭을 통해 칵테일을 검색합니다.
다음 주의사항, 키워드 분류 기준, 예시를 참고하여 키워드를 분류해주세요. 

## 주의사항:
- cocktail: 추론하지 말고, 직접적으로 언급된 칵테일 이름만 (예: "Martini", "Negroni", "Mojito")
- include_ingredients: 포함되어야 하는 재료들 (예: "gin", "whiskey", "lime")
- exclude_ingredients: 제외되어야 하는 재료들 ("없이", "제외", "빼고" 등의 표현으로 언급된 재료들)
- glassType: 글라스 타입들 (예: "martini glass", "highball glass", "rocks glass")
- category: 아래 DB 카테고리 목록에서만 선택
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

## 예시 질문1: "Martini와 비슷한 칵테일을 martini glass에서 마시고 싶어"
→ {{"cocktail": ["Martini"], "include_ingredients": [], "exclude_ingredients": [], "glassType": ["martini glass"], "category": []}}

## 예시 질문2: "gin과 lime이 들어간 highball glass 칵테일"  
→ {{"cocktail": [], "include_ingredients": ["gin", "lime"], "exclude_ingredients": [], "glassType": ["highball glass"], "category": []}}

## 예시 질문3: "whiskey 없이 만드는 rocks glass용 Cocktail 카테고리"
→ {{"cocktail": [], "include_ingredients": [], "exclude_ingredients": ["whiskey"], "glassType": ["rocks glass"], "category": ["Cocktail"]}}

## 예시 질문4: "vodka와 cranberry가 들어간 cocktail glass 칵테일"
→ {{"cocktail": [], "include_ingredients": ["vodka", "cranberry"], "exclude_ingredients": [], "glassType": ["cocktail glass"], "category": []}}

## 예시 질문5: "Negroni 스타일이면서 orange가 포함된 칵테일"
→ {{"cocktail": ["Negroni"], "include_ingredients": ["orange"], "exclude_ingredients": [], "glassType": [], "category": []}}

--------------------------------
사용자 질문: "{user_question}"

Categories: {category_list}

다음 JSON 형식으로만 분류해주세요:
{{
  "cocktail": [],
  "include_ingredients": [],
  "exclude_ingredients": [],
  "glassType": [],
  "category": []
}}

"""

# 시스템 메시지
C2_SYSTEM_MESSAGE = "You are a keyword extraction expert for cocktail search. ONLY extract directly mentioned cocktail names, ingredients, glass types, and categories. Never infer or guess cocktail names from descriptions, colors, or characteristics. Always respond with valid JSON only."

C2_PROMPT_TEMPLATE = """
당신은 글라스 타입과 재료 매칭 전문가입니다.
다음 칵테일 추천 지침과 답변 지침을 고려하여 답변해주세요.

## 칵테일 추천 지침
- 글라스 타입별 칵테일 특성과 서빙 스타일 설명
- 같은 글라스를 사용하는 칵테일들의 공통점과 차이점
- 재료의 레벨별 매칭 (모든 재료 → 일부 재료로 점진적 확장)
- 칵테일 패밀리(Family)별 글라스 사용 패턴

## 답변 지침:
1. 주어진 검색 결과를 바탕으로 정확하게 답변해주세요.
2. 검색 결과에 없는 내용은 추측하지 마세요.
3. 글라스 타입과 재료 조합의 관계를 중점적으로 설명해주세요.
4. 레벨별 재료 매칭 결과를 체계적으로 정리해주세요.

질문: {question}

검색 결과:
{context}

답변:
"""
