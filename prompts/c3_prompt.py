# C3 태스크: Multi-hop 재료 확장 검색
from langchain.prompts import PromptTemplate

# C3 키워드 추출 프롬프트 (Multi-hop 재료 확장용)
C3_KEYWORD_EXTRACTION_PROMPT = """
C3 알고리즘은 Multi-hop 재료 확장 검색을 수행합니다:
1단계: 사용자 재료 → 관련 칵테일들 발견
2단계: 그 칵테일들의 공통 재료들 발견  
3단계: 공통 재료들로 새로운 칵테일들 확장 검색

사용자 질문에서 Multi-hop 재료 확장 검색을 위한 키워드를 추출해주세요.

## 추출할 키워드:
1. ingredients: 언급된 모든 재료 (예: "mint", "lime", "whiskey", "sweet vermouth") 
   - 이 재료들로 시작해서 재료 네트워크를 확장합니다
2. cocktail_names: 구체적인 칵테일 이름 (예: "Manhattan", "Mojito")
   - 이름 유사도로 직접 검색도 병행합니다

## 예시 질문1: "whiskey와 vermouth가 들어간 칵테일과 비슷한 레시피들"
→ {{"ingredients": ["whiskey", "vermouth"], "cocktail_names": []}}

## 예시 질문2: "Manhattan recipe variations and similar cocktails"  
→ {{"ingredients": [], "cocktail_names": ["Manhattan"]}}

## 예시 질문3: "mint와 lime을 사용하는 Mojito 같은 상쾌한 칵테일들"
→ {{"ingredients": ["mint", "lime"], "cocktail_names": ["Mojito"]}}

--------------------------------
사용자 질문: {user_question}

참고 카테고리: {category_list}

다음 JSON 형식으로만 응답해주세요:
{{
    "ingredients": ["언급된 재료들"],
    "cocktail_names": ["구체적인 칵테일 이름들"]
}}
"""

# C3 시스템 메시지
C3_SYSTEM_MESSAGE = "You are a recipe keyword extraction expert. Always respond with valid JSON only."

C3_PROMPT_TEMPLATE = """
당신은 Multi-hop 재료 확장 기반 칵테일 네트워크 전문가입니다.
다음 칵테일 추천 지침과 답변 지침을 고려하여 답변해주세요.

## 칵테일 추천 지침
- 재료 네트워크 확장을 통해 발견된 칵테일들의 연관성 설명
- 초기 재료 → 관련 칵테일 → 공통 재료 → 새로운 칵테일의 확장 과정 설명
- 재료 공유 패턴과 칵테일 패밀리 간의 관계 분석
- 비슷한 재료를 사용하는 칵테일들의 맛 프로필 비교
- 재료 조합의 다양성과 변화 가능성 제시
- 재료별 역할과 대체 가능성 (맛의 변화 포함)
- 확장된 칵테일 네트워크에서 숨겨진 연결고리 발견

## 답변 지침:
1. 주어진 검색 결과를 바탕으로 정확하게 답변해주세요.
2. 검색 결과에 없는 내용은 추측하지 마세요.
3. Multi-hop 확장으로 발견된 칵테일들의 재료 연관성을 체계적으로 설명해주세요.
4. 재료 네트워크를 통한 칵테일 발견 과정과 그 의미를 명확히 제시해주세요.
5. 상세한 레시피 정보와 함께 재료 확장의 논리를 설명해주세요.

질문: {question}

검색 결과:
{context}

답변:
"""
