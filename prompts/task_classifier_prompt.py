from langchain.prompts import PromptTemplate

TASK_CLASSIFIER_TEMPLATE = """
당신은 칵테일 질문 분류 전문가입니다. 사용자의 질문을 분석하여 다음 4가지 태스크 중 하나로 정확하게 분류해주세요.

**태스크 정의:**

**C1: 색상-재료 기반 시각 검색**
- 색상 키워드가 포함된 질문 ("red", "blue", "golden", "green" 등)
- 시각적 외관, 유사한, 비슷한 칵테일
- "elegant appearance", "beautiful", "visually appealing" 등 시각적 형용사
- 색상과 재료의 연관성 질문
- 색상 대비나 시각적 조화 관련

**C2: Glass Type + 재료 매칭**
- 특정 글라스 타입이 명시된 질문 ("highball glass", "martini glass", "rocks glass" 등)
- 여러 재료 조합과 글라스의 관계
- 글라스에 따른 칵테일 추천
- "served in", "glass", "tumbler" 등 글라스 관련 키워드
- 칵테일 + 글라스 + 재료의 복합 조건

**C3: Multi-hop 재료 확장 검색**
- **재료 리스트가 주어진** 질문 (칵테일 이름 없이)
- "whiskey와 vermouth로 만들 수 있는 칵테일"
- 재료 → 칵테일 → 공통재료 → 새로운 칵테일 탐색
- "이 재료들로 뭘 만들 수 있나요?"
- 재료 조합 기반 발견적 검색

**C4: 칵테일 레시피 유사도 및 대안 추천**
- **특정 칵테일 이름이 타겟으로 명시된** 질문
- "Manhattan과 레시피가 유사한", "Mojito 같은 레시피 "
- "Old Fashioned 대신 마실만한"
- 타겟 칵테일과 공유 재료가 많은 대안
- 비슷한 복잡도(재료 개수)의 칵테일 추천


**분류 예시:**

**C1 예시:**
- "red layered cocktail with elegant appearance" → C1 (색상 + 시각적 특징)
- "golden colored cocktail with whiskey base" → C1 (색상 키워드 중심)
- "blue tropical drink similar to ocean colors" → C1 (색상 기반 시각 검색)

**C2 예시:**
- "gin과 lime이 들어간 highball glass 칵테일" → C2 (글라스 + 재료 조합)
- "Martini와 비슷한 칵테일을 martini glass에서 마시고 싶어" → C2 (글라스 중심)
- "vodka와 cranberry가 들어간 cocktail glass 칵테일" → C2 (글라스 + 재료)

**C3 예시:**
- "whiskey와 vermouth가 들어간 칵테일들" → C3 (재료 리스트만 주어짐)
- "mint와 lime을 사용하는 칵테일 찾아줘" → C3 (재료 조합→칵테일 발견)
- "vodka와 cranberry로 뭘 만들 수 있나요?" → C3 (재료→칵테일 탐색)

**C4 예시:**
- "Manhattan과 유사한 레시피의 칵테일 추천해줘" → C4 (타겟=Manhattan 명시)
- "Mojito 같은 스타일의 다른 레시피의 칵테일들" → C4 (타겟=Mojito 기준)
- "Old Fashioned 대신 마실만한 레시피의 칵테일" → C4 (타겟=Old Fashioned 대안)

**답변 형식:**
반드시 다음 JSON 형식으로만 답변하세요:
{{
    "task": "C1",
    "confidence": 85,
    "reason": "시각적 유사성과 추천을 요청하는 질문이므로 C1이 적합"
}}

질문: {question}

분류:
"""

# 태스크 분류를 위한 프롬프트
task_classifier_prompt = PromptTemplate(
    template=TASK_CLASSIFIER_TEMPLATE,
    input_variables=["question"]
)