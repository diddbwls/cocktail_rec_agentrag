# C1 태스크: 시각 및 재료 기반 검색

from langchain.prompts import PromptTemplate
#C1 키워드 추출 프롬프트 - Visual Similarity 기반 검색
C1_KEYWORD_EXTRACTION_PROMPT = """
C1 알고리즘은 시각 및 재료 기반 검색을 수행합니다.
다음 주의사항, 재료 분류 기준, 예시를 참고하여 키워드를 분류해주세요. 

## 주의사항:
- cocktail: 추론하지 말고, 직접적으로 언급된 칵테일 이름만 (예: "Martini", "Negroni")
- include_ingredients: 포함되어야 하는 재료들 (예: "cherry", "vodka")
- exclude_ingredients: 제외되어야 하는 재료들 ("없이", "제외", "빼고" 등의 표현으로 언급된 재료들)
- glassType: 글라스 타입들 (예: "martini glass", "highball glass")
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
- visual_keywords: 색상 키워드들만 (예: "red", "blue", "golden", "orange", "green", "purple", "pink", "yellow", "black", "white", "brown")

## 재료 분류 기준:
- "cherry가 들어간", "cherry 필수" → include_ingredients에 "cherry"
- "vodka 없이", "vodka 제외하고", "vodka 빼고" → exclude_ingredients에 "vodka"
- "gin이 있으면서 tonic 없는" → include_ingredients에 "gin", exclude_ingredients에 "tonic"

## 예시 질문1: "red cocktail with cherry garnish"
→ {{"include_ingredients": [], "exclude_ingredients": [], "cocktail": [], "glassType": [], "category": [], "visual_keywords": ["red"]}}

## 예시 질문2: "blue tropical drink similar to ocean colors"  
→ {{"include_ingredients": [], "exclude_ingredients": [], "cocktail": [], "glassType": [], "category": [], "visual_keywords": ["blue"]}}

## 예시 질문3: "golden colored cocktail with whiskey base"
→ {{"include_ingredients": ["whiskey"], "exclude_ingredients": [], "cocktail": [], "glassType": [], "category": [], "visual_keywords": ["golden"]}}

## 예시 질문4: "Manhattan과 비슷한 red colored cocktail" 
→ {{"include_ingredients": [], "exclude_ingredients": [], "cocktail": ["Manhattan"], "glassType": [], "category": [], "visual_keywords": ["red"]}}


--------------------------------
사용자 질문: "{user_question}"

Categories: {category_list}

다음 JSON 형식으로만 분류해주세요:
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
당신은 시각적 칵테일 검색 전문가입니다.
다음 칵테일 추천 지침과 답변 지침을 고려하여 답변해주세요.

## 칵테일 추천 지침
- 색상과 시각적 특성을 중심으로 한 추천
- 칵테일의 외관(appearance), 색상(color), 레이어링(layering) 등 시각적 요소 강조
- 비슷한 색상이나 시각적 스타일을 가진 칵테일들의 그룹핑
- 재료의 색상이 칵테일 외관에 미치는 영향 설명
- 계절별, 상황별 색상 추천 (여름=밝고 시원한 색, 겨울=따뜻하고 진한 색)
- 레이어드 칵테일과 그라데이션 효과 설명

## 답변 지침:
1. 주어진 검색 결과를 바탕으로 정확하게 답변해주세요.
2. 검색 결과에 없는 내용은 추측하지 마세요.
3. 칵테일의 시각적 특성(색상, 외관, 레이어링)을 중점적으로 설명해주세요.
4. 색상 재료와 칵테일 외관의 연관성을 명확히 설명해주세요.

질문: {question}

검색 결과:
{context}

답변:
"""