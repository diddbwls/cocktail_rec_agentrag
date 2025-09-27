from typing import Dict, Any, List
import sys
import os

# 상위 디렉토리의 모듈 임포트를 위한 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.openai_client import OpenAIClient
from prompts.c1_prompt import C1_PROMPT_TEMPLATE
from prompts.c2_prompt import C2_PROMPT_TEMPLATE
from prompts.c3_prompt import C3_PROMPT_TEMPLATE
from prompts.c4_prompt import C4_PROMPT_TEMPLATE

def format_system_analysis_info(task_type: str, task_confidence: float, task_reason: str,
                               final_best_round: int, final_best_score: float, 
                               final_best_top_k: int, cocktails_count: int,
                               evaluation_scores: Dict[str, float], 
                               reflection_feedback: str) -> str:
    """
    format system analysis info for user-friendly
    
    Args:
        task_type: 분류된 태스크 타입
        task_confidence: 분류 신뢰도
        task_reason: 분류 이유
        final_best_round: 선택된 라운드
        final_best_score: 최고 점수
        final_best_top_k: 선택된 때의 Top-K
        cocktails_count: 추천된 칵테일 수
        evaluation_scores: 평가 점수들
        reflection_feedback: 리플렉션 피드백
        
    Returns:
        포맷된 시스템 분석 정보 텍스트
    """
    # 태스크 타입별 이름 매핑
    task_names = {
        "C1": "Color-Ingredient Visual Search",
        "C2": "Glass Type + Ingredient Matching", 
        "C3": "Multi-hop Ingredient Expansion Search",
        "C4": "Cocktail Recipe Similarity and Alternative Recommendation"
    }
    
    task_name = task_names.get(task_type, task_type)
    
    # classification info
    classification_info = f"""🎯 task classification:
- classification result: {task_type} ({task_name})
- confidence: {task_confidence:.1f}%
- classification reason: {task_reason}"""
    
    # selection info - round-wise situation description
    if final_best_round == 1:
        selection_reason = "첫 번째 검색에서 충분한 품질 달성"
    elif final_best_score >= 80:
        selection_reason = f"Round {final_best_round}에서 품질 기준(80점) 달성"
    else:
        selection_reason = f"3회 반복 중 Round {final_best_round}에서 최고 품질 달성"
    
    selection_info = f"""🏆 final selection:
- selected round: Round {final_best_round}
- recommended cocktails: {cocktails_count} (Top-{final_best_top_k} search)
- quality score: {final_best_score:.1f}/100 points
- selection reason: {selection_reason}"""
    
    # 평가 점수 - 더 상세한 설명
    if evaluation_scores:
        try:
            relevance = float(evaluation_scores.get('relevance', 0))
            diversity = float(evaluation_scores.get('diversity', 0))
            completeness = float(evaluation_scores.get('completeness', 0))
            coherence = float(evaluation_scores.get('coherence', 0))
            
            evaluation_info = f"""📊 evaluation details:
- relevance (Relevance): {relevance:.1f}/100 points - relevance to the question
- diversity (Diversity): {diversity:.1f}/100 points - recommendation diversity
- completeness (Completeness): {completeness:.1f}/100 points - requirement satisfaction
- coherence (Coherence): {coherence:.1f}/100 points - logical consistency
- overall score: {final_best_score:.1f}/100 points

💡 system feedback: {reflection_feedback}"""
        except (ValueError, TypeError):
            evaluation_info = f"""📊 evaluation:
- overall score: {final_best_score:.1f}/100 points
- feedback: {reflection_feedback}"""
    else:
        evaluation_info = f"""📊 evaluation:
- overall score: {final_best_score:.1f}/100 points"""
    
    return f"""
{classification_info}

{selection_info}

{evaluation_info}
"""

# OpenAI 클라이언트 초기화
openai_client = OpenAIClient()

def format_cocktails_for_response(cocktails: List[Dict[str, Any]]) -> str:
    """
    format cocktails list for final response
    
    Args:
        cocktails: cocktails info list
        
    Returns:
        포맷된 텍스트
    """
    if not cocktails:
        return "No cocktails found."
    
    formatted_lines = []
    
    for i, cocktail in enumerate(cocktails, 1):
        name = cocktail.get('name', 'Unknown')
        category = cocktail.get('category', 'N/A')
        glass_type = cocktail.get('glassType', 'N/A')
        alcoholic = cocktail.get('alcoholic', 'N/A')
        description = cocktail.get('description', '')
        instructions = cocktail.get('instructions', '')
        
        # header
        formatted_lines.append(f"{i}. **{name}**")
        formatted_lines.append(f"   - category: {category}")
        formatted_lines.append(f"   - glass_type: {glass_type}")
        formatted_lines.append(f"   - alcoholic: {alcoholic}")
        
        # ingredients info
        recipe_ingredients = cocktail.get('recipe_ingredients', [])
        ingredients = cocktail.get('ingredients', [])
        
        if recipe_ingredients:
            formatted_lines.append("   - ingredients:")
            for ingredient_info in recipe_ingredients:
                measure = ingredient_info.get('measure', 'unknown')
                ingredient = ingredient_info.get('ingredient', 'unknown')
                formatted_lines.append(f"     • {measure} {ingredient}")
        elif ingredients:
            formatted_lines.append(f"   - ingredients: {', '.join(ingredients)}")
        
        # instructions
        if instructions:
            formatted_lines.append(f"   - instructions: {instructions}")
            
        # description
        if description:
            formatted_lines.append(f"   - description: {description}")
        
        formatted_lines.append("")  # empty line
    
    return "\n".join(formatted_lines)

def generate_final_response(user_query: str, cocktails: List[Dict[str, Any]], 
                          task_type: str, evaluation_scores: Dict[str, float],
                          reflection_feedback: str, task_confidence: float = 0,
                          task_reason: str = "", final_best_round: int = 1,
                          final_best_score: float = 0, final_best_top_k: int = 3) -> str:
    """
    최종 응답 생성
    
    Args:
        user_query: 사용자 질문
        cocktails: 추천 칵테일 리스트
        task_type: 태스크 타입 (C1-C4)
        evaluation_scores: 평가 점수들
        reflection_feedback: 리플렉션 피드백
        
    Returns:
        최종 응답 텍스트
    """
    try:
        # 태스크별 프롬프트 직접 사용 (응답 생성용)
        task_prompts = {
            "C1": C1_PROMPT_TEMPLATE,
            "C2": C2_PROMPT_TEMPLATE,
            "C3": C3_PROMPT_TEMPLATE,
            "C4": C4_PROMPT_TEMPLATE
        }
        task_prompt = task_prompts.get(task_type, C1_PROMPT_TEMPLATE)
        
        # 디버깅: 태스크 프롬프트 확인
        print(f"🔍 태스크 프롬프트 확인 ({task_type}):")
        print(f"   - 프롬프트 길이: {len(task_prompt)} 글자")
        print(f"   - context 플레이스홀더 포함: {'context' in task_prompt}")
        print(f"   - question 플레이스홀더 포함: {'question' in task_prompt}")
        if 'context' in task_prompt:
            # context가 어디에 위치하는지 확인
            context_pos = task_prompt.find('{context}')
            context_preview = task_prompt[max(0, context_pos-50):context_pos+100] if context_pos != -1 else "NOT_FOUND"
            print(f"   - context 위치 주변: ...{context_preview}")
        
        # 칵테일 정보 포맷팅
        cocktails_context = format_cocktails_for_response(cocktails)
        
        # 디버깅: 칵테일 컨텍스트 내용 확인
        print(f"🔍 칵테일 컨텍스트 확인:")
        print(f"   - 칵테일 수: {len(cocktails)}")
        if cocktails:
            cocktail_names = [c.get('name', 'Unknown') for c in cocktails]
            print(f"   - 칵테일 이름들: {cocktail_names}")
        print(f"   - 포맷된 컨텍스트 길이: {len(cocktails_context)} 글자")
        print(f"   - 컨텍스트 미리보기 (처음 200자): {cocktails_context}")
        
        # 평가 정보 포맷팅
        evaluation_text = ""
        if evaluation_scores:
            # 각 점수를 float로 변환하여 안전하게 계산
            try:
                relevance = float(evaluation_scores.get('relevance', 0))
                diversity = float(evaluation_scores.get('diversity', 0))
                completeness = float(evaluation_scores.get('completeness', 0))
                coherence = float(evaluation_scores.get('coherence', 0))
                
                overall_score = (relevance + diversity + completeness + coherence) / 4
                
                evaluation_text = f"""
evaluation scores:
- relevance: {relevance:.1f} points
- diversity: {diversity:.1f} points  
- completeness: {completeness:.1f} points
- coherence: {coherence:.1f} points
- overall: {overall_score:.1f} points

{reflection_feedback}
"""
            except (ValueError, TypeError) as e:
                print(f"⚠️ 평가 점수 변환 오류: {e}")
                evaluation_text = f"""
Evaluation scores conversion error occurred.
original data: {evaluation_scores}

{reflection_feedback}
"""
        
        # 프롬프트에 정보 삽입
        prompt = task_prompt.format(
            question=user_query,
            context=cocktails_context
        )
        
        # 시스템 분석 정보 생성
        system_analysis = format_system_analysis_info(
            task_type=task_type,
            task_confidence=task_confidence,
            task_reason=task_reason,
            final_best_round=final_best_round,
            final_best_score=final_best_score,
            final_best_top_k=final_best_top_k,
            cocktails_count=len(cocktails),
            evaluation_scores=evaluation_scores,
            reflection_feedback=reflection_feedback
        )

        # 순수한 칵테일 정보만으로 프롬프트 구성 (시스템 분석 정보 제외)
        enhanced_prompt = f"""{prompt}

Based on the cocktail information above, please recommend cocktails with detailed explanations that are helpful for the user.
Include reasons for recommendation, flavor characteristics, and situational suggestions in your explanation."""
        print(f"🎯 최종 응답 생성 중... ({task_type})")
        
        # LLM이 받는 최종 컨텍스트를 HTML로 표시
        from IPython.display import display, HTML
        
        # HTML 형태로 컨텍스트 포맷팅
        html_content = f"""
        <div style="border: 2px solid #4CAF50; border-radius: 10px; padding: 20px; margin: 10px 0; background-color: #f8f9fa;">
            <h3 style="color: #2E8B57; margin-top: 0;">🤖 LLM이 받는 최종 컨텍스트 ({task_type})</h3>
            <div style="background-color: white; border: 1px solid #ddd; border-radius: 5px; padding: 15px; font-family: monospace; white-space: pre-wrap; max-height: 600px; overflow-y: auto;">
{enhanced_prompt}
            </div>
        </div>
        """
        
        display(HTML(html_content))
        
        # OpenAI API 호출
        response = openai_client.generate(enhanced_prompt, max_tokens=1500)
        
        return response
        
    except Exception as e:
        print(f"❌ 응답 생성 오류: {e}")
        # 기본 응답 생성
        cocktails_text = format_cocktails_for_response(cocktails)
        return f"""Sorry, an error occurred while generating the response.

We recommend the following cocktails:

{cocktails_text}

오류 내용: {str(e)}"""

def generator(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    비교용 응답을 생성하는 Generator 노드
    
    초기 답변과 최종 답변을 각각 생성하여 비교할 수 있도록 합니다.
    
    Args:
        state: 파이프라인 상태 딕셔너리
        
    Returns:
        초기 답변과 최종 답변이 포함된 상태 딕셔너리
    """
    # 필요한 정보 추출
    user_query = state.get("input_text_with_image", state.get("input_text", ""))
    task_type = state.get("task_type", "C1")
    iteration_count = state.get("iteration_count", 0)
    
    print(f"\n📝 비교용 응답 생성 시작 ({task_type}, {iteration_count}회 반복)")
    
    # 1. 초기 답변 생성 (Round 1 결과 사용)
    initial_results = state.get("initial_search_results", [])
    if initial_results:
        print(f"🥉 초기 답변 생성 중... ({len(initial_results)}개 칵테일)")
        try:
            # 태스크 분류 정보 가져오기
            task_confidence = state.get("task_confidence", 0)
            task_reason = state.get("task_reason", "정보 없음")
            
            # 초기 평가 점수 사용 (reflection에서 저장한 것)
            initial_evaluation_scores = state.get("initial_evaluation_scores", {})
            initial_score = state.get("initial_score", 0)
            initial_feedback = state.get("initial_feedback", "초기 검색 결과")
            
            # 초기 점수가 없는 경우, best_result에서 Round 1 정보 찾기
            if initial_score == 0 and "best_result" in state:
                best_result = state["best_result"]
                if best_result.get("iteration") == 1:
                    # best_result가 Round 1 결과인 경우
                    initial_score = best_result.get("score", 0)
                    initial_evaluation_scores = best_result.get("evaluation", {})
                    initial_feedback = initial_evaluation_scores.get("feedback", "Round 1 검색 결과")
                    print(f"📋 best_result에서 Round 1 점수 복구: {initial_score}점")
                else:
                    # 디버깅을 위해 debug_info에서 reflection 히스토리 확인
                    reflection_history = state.get("debug_info", {}).get("reflection_history", [])
                    if reflection_history:
                        round1_result = reflection_history[0]  # 첫 번째 라운드
                        initial_score = round1_result.get("score", 0)
                        initial_evaluation_scores = round1_result.get("scores", {})
                        initial_feedback = initial_evaluation_scores.get("feedback", "Round 1 검색 결과")
                        print(f"📋 reflection_history에서 Round 1 점수 복구: {initial_score}점")
            
            print(f"🔍 초기 평가 점수 (최종 확인):")
            print(f"   - initial_evaluation_scores: {initial_evaluation_scores}")
            print(f"   - initial_score: {initial_score}")
            print(f"   - initial_feedback: {initial_feedback}")
            
            initial_response = generate_final_response(
                user_query=user_query,
                cocktails=initial_results,
                task_type=task_type,
                evaluation_scores=initial_evaluation_scores,
                reflection_feedback=initial_feedback,
                task_confidence=task_confidence,
                task_reason=task_reason,
                final_best_round=1,
                final_best_score=initial_score,
                final_best_top_k=len(initial_results)
            )
            state["initial_response"] = initial_response
            print(f"✅ 초기 답변 생성 완료")
        except Exception as e:
            print(f"❌ 초기 답변 생성 오류: {e}")
            state["initial_response"] = f"초기 답변 생성 중 오류가 발생했습니다: {str(e)}"
    
    # 2. 최종 답변 생성 (최고 점수 라운드 결과 사용)
    final_results = state.get("final_search_results", [])
    if final_results:
        final_best_round = state.get("final_best_round", 3)
        final_best_score = state.get("final_best_score", 0)
        final_best_top_k = state.get("final_best_top_k", len(final_results))
        
        print(f"🏆 최종 답변 생성 중... (Round {final_best_round}, {len(final_results)}개 칵테일, {final_best_score:.1f}점)")
        
        evaluation_scores = state.get("evaluation_scores", {})
        reflection_feedback = state.get("reflection_feedback", "")
        
        try:
            # 태스크 분류 정보 가져오기
            task_confidence = state.get("task_confidence", 0)
            task_reason = state.get("task_reason", "정보 없음")
            
            final_response = generate_final_response(
                user_query=user_query,
                cocktails=final_results,
                task_type=task_type,
                evaluation_scores=evaluation_scores,
                reflection_feedback=reflection_feedback,
                task_confidence=task_confidence,
                task_reason=task_reason,
                final_best_round=final_best_round,
                final_best_score=final_best_score,
                final_best_top_k=final_best_top_k
            )
            state["final_response"] = final_response
            state["final_text"] = final_response  # 하위 호환성
            state["final_cocktails"] = final_results  # 하위 호환성
            print(f"✅ 최종 답변 생성 완료")
        except Exception as e:
            print(f"❌ 최종 답변 생성 오류: {e}")
            state["final_response"] = f"최종 답변 생성 중 오류가 발생했습니다: {str(e)}"
            state["final_text"] = f"최종 답변 생성 중 오류가 발생했습니다: {str(e)}"
            state["final_cocktails"] = final_results
    
    # 3. 응답이 없는 경우 기본 처리
    if not initial_results and not final_results:
        print("❌ 검색 결과가 없어 기본 응답을 생성합니다.")
        error_response = f"""죄송합니다. "{user_query}"에 대한 적절한 칵테일을 찾지 못했습니다.

다시 한번 다른 키워드나 설명으로 질문해주시거나, 다음과 같이 구체적으로 질문해보세요:
- 선호하는 색상이나 맛
- 특정 재료나 베이스 술
- 마시고 싶은 상황이나 분위기
- 글라스 타입이나 스타일"""
        
        state["initial_response"] = error_response
        state["final_response"] = error_response
        state["final_text"] = error_response
        state["final_cocktails"] = []
    
    # 디버그 정보 추가
    if "debug_info" not in state:
        state["debug_info"] = {}
    
    state["debug_info"]["generation"] = {
        "task_type": task_type,
        "initial_cocktails_count": len(initial_results),
        "final_cocktails_count": len(final_results),
        "initial_response_length": len(state.get("initial_response", "")),
        "final_response_length": len(state.get("final_response", "")),
        "iteration_count": iteration_count,
        "final_best_round": state.get("final_best_round", 0),
        "final_best_score": state.get("final_best_score", 0),
        "final_best_top_k": state.get("final_best_top_k", 0)
    }
    
    return state