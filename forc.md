본 연구에서는 칵테일 모달리티 정보를 입력으로 받아 태스크에 맞게 추천하는 프레임워크를 제안한다. 
이 과정에서 task recognition과 reflection 단계의 핵심 과정을 거치며, 특히 Graph DB를 구축하고 DB 정보에서 가장 적절한 context를 도출하기 위해 reflection을 한다. 
두 단계 모두 LLM이 진행하며, 아래는 Graph DB 구축, agent(task recognition,Reflection), graph retrievald에서의 각 단계를 상세히 설명한다.

# 구축 주의사항
- 이 프로젝트는 langgraph 기반이므로 pipeline.py의 노드 경계는 dict타입으로 통일할 것
- pipeline에 노드추가가 필요하면 추가하되, 어떤 노드가 추가되었는지 확실히 명시할 것
- 본 프로젝트는 langgraph pipeline 따로, retrieval 알고리즘 따로 구축되었기 때문에 매끄러운 integration이 반드시 필요한 프로젝트임
- 일부 qwen으로 테스트한 흔적이 있을 수 있으나, 해당 흔적은 정리하고, openai llm모델을 사용할 것(/Users/yujin/Desktop/cocktail_rec_agentrag/langgraph/util/config.py 참조)
- user.ipynb에서 langgraph를 c1~c4 task별로 1개씩 최종 테스트할 수 있게 할 것.
- 유지보수가 가능하도록 주석을 남길 것

# 1. Graph DB 구축
Graph Structure
--> 완료 (cocktail_graph_builder.py)

# 2. LLM as a Agent: Task-recognition & Reflection
본 연구에서 사용되는 Agent는 Task-recognition router와 reflection 두 가지이다. 
task recognition은 llm이 사용자의 질문 의도에 맞게 각 검색 방법을 정하는 router이다. 
reflection은 검색결과를 바탕으로 llm이 판단하여 검색 결과의 품질을 평가한 뒤, 필요 시 top-k 값을 조정하여 후보군을 확장할지를 결정하는 과정이다. 각 agent별 상세한 설명은 다음과 같다. 

**2.1 Task recognition**

관련 코드 
- /Users/yujin/Desktop/cocktail_rec_agentrag/langgraph/retrieval
- /Users/yujin/Desktop/cocktail_rec_agentrag/langgraph/prompts

Task recognition Router는 사용자의 자연어 질문을 분석하여 4개의 정의된 classification 태스크(C1~C4) 중 가장 적합한 태스크로 분류하는 역할을 수행한다. 각 태스크는 칵테일 추천의 특정 측면에 최적화된 검색 알고리즘을 가지고 있으며, 다음과 같이 정의된다:

- C1: 색상-ingredient 시각적 검색 알고리즘
- C2: glasstype-ingredient 매칭
- C3: multi-hop ingredient 확장 검색
- C4: 재료기반 유사 레시피 칵테일 추천

Task Classifier는 LLM을 활용하여 사용자 질문의 의도와 키워드를 분석하고, JSON 형식으로 태스크 분류 결과와 신뢰도(confidence), 분류 이유(reason)를 반환한다. 이를 통해 각 질문에 가장 적합한 검색 알고리즘이 자동으로 선택되어 실행된다.

이 중 ../prompts/query_image_prompt.py로 image를 description한 text는, ../langgraph/user.ipynb 질문할때 사용자가 사진의 path를 주면, image description이 이루어지도록 연동 필요

**2.2 Reflection**

Reflection은 검색된 칵테일 후보들의 품질을 평가하고 최적의 추천 결과를 선택하는 반성적 추론 과정이다. Reflection의 과정은 다음과 같다:

목적
동일 질의에 대해 서로 다른 k(검색 상위 개수)를 단계적으로 시도한다. 각 k마다 완결된 답변을 새로 생성하고, 품질 임계값을 만족하면 즉시 채택(early accept) 한다. 전 단계가 모두 실패하면 그중 가장 우수한 답변을 보정하여 최종 제시한다.

반복 규칙 --> pipeline.py 수정이 필요하면 진행
최대 반복 횟수: 3회
k는 매 라운드마다 1씩 증가 (top-k → top-(k+1) → top-(k+2))
각 라운드: 검색 → 답변 생성 → 평가 → 임계 검증

라운드 절차
1. **Candidate Generation**: 각 검색 알고리즘에서 top-k후보를 생성한다. llm에 현재 k 후보의 정보를 context로 전달하여 답변을 생성한다. 
2. **Quality Assessment**: LLM Agent가 각 후보 세트를 평가한다. 평가 기준은 다음 4항목을 포함한다
모든 항목이 임계 이상이면 → 즉시 답변 확정 후 종료 :
    - 사용자 질문과의 관련성 (relevance score)
    - 추천 결과의 다양성 (diversity score)
    - 정보의 완전성 (completeness score)
    - 설명의 일관성 (coherence score)
3. **Iterative Refinement**: 설정된 품질 임계값을 만족하지 못할 경우, 미달항목과 사유를 기록한다. 그 후
검색 파라미터인 top-k 값을 점진적으로 확장(top-k, top-(k+1), top-(k+2), …)하여
재검색을 수행한다. 이 과정은 최대 3회까지 반복한다.
4. **Final Selection**: 3회까지 반복했음에도 결과를 확정짓지 못했을 경우, 후보 세트 중에서 종합 점수가 가장 높은 결과를 최종적으로 선택한다. Agent는 이 답변이 부족한 이유와 함께 사용자에게 추천 결과를 제시한다. 
예: “이 답변은 관련성/완전성 점수는 높지만, 다양성이 부족했습니다.”
만약 동점 항목이 있을경우, relevance score>diversity score>completeness score>coherence score 로 순서를 구분한다.


# 3. Graph RAG : **Task-Specific Retrieval Methods**

/Users/yujin/Desktop/cocktail_rec_agentrag/langgraph/retrieval에 각각 c*_retrieval.py로 아래 retrieval 알고리즘이 작성되어있다. 출력형식을 바꿔야할 경우 수정해도 괜찮지만, 절대 핵심적인 알고리즘을 수정해서는 안된다. 
- C1: 색상-ingredient 시각적 검색 알고리즘
- C2: glasstype-ingredient 매칭
- C3: multi-hop ingredient 확장 검색
- C4: 재료기반 유사 레시피 칵테일 추천
