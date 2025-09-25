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
C4 알고리즘은 관계 기반 + 레시피 복잡도 기반 칵테일 대안을 검색합니다:
- 타겟 칵테일과 공유 재료가 많은 칵테일들 우선 선정
- 레시피 복잡도(재료 개수)가 비슷한 칵테일들로 필터링
- 그래프 관계(HAS_INGREDIENT)를 통한 유사도 계산

사용자 질문에서 칵테일 유사도 검색을 위한 타겟 정보를 추출해주세요.

## 추출할 정보:
1. target_cocktail: 기준이 되는 칵테일 이름 (예: "Manhattan", "Mojito", "Martini")
   - 가장 중요한 요소! 이것이 있어야 관계 기반 검색 가능
2. ingredients: 언급된 구체적인 재료들 (예: "whiskey", "vermouth", "bitters")  
   - target_cocktail이 없을 때 재료로 타겟을 찾는 용도

## 예시 질문1: "Manhattan과 유사한 칵테일 추천해줘"
→ {{"target_cocktail": "Manhattan", "ingredients": []}}

## 예시 질문2: "Mojito 같은 스타일의 다른 칵테일들"  
→ {{"target_cocktail": "Mojito", "ingredients": []}}

## 예시 질문3: "whiskey와 sweet vermouth를 사용하는 칵테일과 비슷한 레시피"
→ {{"target_cocktail": "", "ingredients": ["whiskey", "sweet vermouth"]}}

## 예시 질문4: "Old Fashioned와 비슷하지만 다른 재료를 사용하는 칵테일"
→ {{"target_cocktail": "Old Fashioned", "ingredients": []}}

--------------------------------
사용자 질문: {user_question}

참고 카테고리: {category_list}

다음 JSON 형식으로만 응답해주세요:
{{
    "target_cocktail": "기준 칵테일 이름 (없으면 빈 문자열)",
    "ingredients": ["언급된 재료들"]
}}
"""

# C4 시스템 메시지
C4_SYSTEM_MESSAGE = "You are a cocktail similarity expert. Always respond with valid JSON only."

C4_PROMPT_TEMPLATE = """
당신은 칵테일 유사도 및 대안 추천 전문가입니다.
다음 칵테일 추천 지침과 답변 지침을 고려하여 답변해주세요.

## 칵테일 추천 지침
- 레시피 복잡도(재료 개수)가 비슷한 칵테일 우선 추천
- 베이스 스피릿이 같은 칵테일 강조
- 제조 기법의 유사성 설명 (셰이킹 vs 스티어링)
- 맛 프로필의 유사점과 차이점 비교
- 대체 재료 제안 시 맛의 변화 설명

## 답변 지침:
1. 주어진 검색 결과를 바탕으로 정확하게 답변해주세요.
2. 검색 결과에 없는 내용은 추측하지 마세요.
3. 타겟 칵테일과의 유사점과 차이점을 명확히 설명해주세요.
4. 재료 공유도와 복잡도를 언급하여 추천 이유를 설명해주세요.

질문: {question}

검색 결과:
{context}

답변:
"""