from typing import Dict, Any, List
import sys
import os

# 상위 디렉토리의 모듈 임포트를 위한 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.openai_client import OpenAIClient
from utils.prompt_loader import PromptLoader

def format_system_analysis_info(task_type: str, task_confidence: float, task_reason: str,
                               final_best_round: int, final_best_score: float, 
                               final_best_top_k: int, cocktails_count: int,
                               evaluation_scores: Dict[str, float], 
                               reflection_feedback: str) -> str:
    """
    시스템 분석 정보를 사용자 친화적으로 포맷팅
    
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
        "C1": "색상-재료 기반 시각 검색",
        "C2": "글라스 타입 + 재료 매칭", 
        "C3": "Multi-hop 재료 확장 검색",
        "C4": "칵테일 유사도 및 대안 추천"
    }
    
    task_name = task_names.get(task_type, task_type)
    
    # 분류 정보
    classification_info = f"""🎯 태스크 분류:
- 분류 결과: {task_type} ({task_name})
- 신뢰도: {task_confidence:.1f}%
- 분류 이유: {task_reason}"""
    
    # 선택 정보 - 라운드별 상황 설명
    if final_best_round == 1:
        selection_reason = "첫 번째 검색에서 충분한 품질 달성"
    elif final_best_score >= 80:
        selection_reason = f"Round {final_best_round}에서 품질 기준(80점) 달성"
    else:
        selection_reason = f"3회 반복 중 Round {final_best_round}에서 최고 품질 달성"
    
    selection_info = f"""🏆 최종 선택:
- 선택된 라운드: Round {final_best_round}
- 추천 칵테일 수: {cocktails_count}개 (Top-{final_best_top_k} 검색)
- 품질 점수: {final_best_score:.1f}/100점
- 선택 이유: {selection_reason}"""
    
    # 평가 점수 - 더 상세한 설명
    if evaluation_scores:
        try:
            relevance = float(evaluation_scores.get('relevance', 0))
            diversity = float(evaluation_scores.get('diversity', 0))
            completeness = float(evaluation_scores.get('completeness', 0))
            coherence = float(evaluation_scores.get('coherence', 0))
            
            evaluation_info = f"""📊 품질 평가 세부 점수:
- 관련성 (Relevance): {relevance:.1f}/100점 - 질문과의 연관성
- 다양성 (Diversity): {diversity:.1f}/100점 - 추천의 다양성
- 완전성 (Completeness): {completeness:.1f}/100점 - 요구사항 충족도
- 일관성 (Coherence): {coherence:.1f}/100점 - 논리적 일관성
- 전체 점수: {final_best_score:.1f}/100점

💡 시스템 피드백: {reflection_feedback}"""
        except (ValueError, TypeError):
            evaluation_info = f"""📊 품질 평가:
- 전체 점수: {final_best_score:.1f}/100점
- 피드백: {reflection_feedback}"""
    else:
        evaluation_info = f"""📊 품질 평가:
- 전체 점수: {final_best_score:.1f}/100점"""
    
    return f"""
{classification_info}

{selection_info}

{evaluation_info}
"""

# OpenAI 클라이언트 초기화
openai_client = OpenAIClient()
prompt_loader = PromptLoader()

def format_cocktails_for_response(cocktails: List[Dict[str, Any]]) -> str:
    """
    칵테일 리스트를 최종 응답용으로 포맷팅
    
    Args:
        cocktails: 칵테일 정보 리스트
        
    Returns:
        포맷된 텍스트
    """
    if not cocktails:
        return "추천할 칵테일을 찾지 못했습니다."
    
    formatted_lines = []
    
    for i, cocktail in enumerate(cocktails, 1):
        name = cocktail.get('name', 'Unknown')
        category = cocktail.get('category', 'N/A')
        glass_type = cocktail.get('glassType', 'N/A')
        alcoholic = cocktail.get('alcoholic', 'N/A')
        description = cocktail.get('description', '')
        instructions = cocktail.get('instructions', '')
        
        # 헤더
        formatted_lines.append(f"{i}. **{name}**")
        formatted_lines.append(f"   - 카테고리: {category}")
        formatted_lines.append(f"   - 글라스 타입: {glass_type}")
        formatted_lines.append(f"   - 알코올: {alcoholic}")
        
        # 재료 정보
        recipe_ingredients = cocktail.get('recipe_ingredients', [])
        ingredients = cocktail.get('ingredients', [])
        
        if recipe_ingredients:
            formatted_lines.append("   - 재료:")
            for ingredient_info in recipe_ingredients:
                measure = ingredient_info.get('measure', 'unknown')
                ingredient = ingredient_info.get('ingredient', 'unknown')
                formatted_lines.append(f"     • {measure} {ingredient}")
        elif ingredients:
            formatted_lines.append(f"   - 재료: {', '.join(ingredients)}")
        
        # 제조법
        if instructions:
            formatted_lines.append(f"   - 제조법: {instructions}")
            
        # 설명
        if description:
            formatted_lines.append(f"   - 설명: {description}")
        
        formatted_lines.append("")  # 빈 줄
    
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
        # 태스크별 프롬프트 로드
        task_prompt = prompt_loader.get_task_prompt(task_type)
        
        # 칵테일 정보 포맷팅
        cocktails_context = format_cocktails_for_response(cocktails)
        
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
평가 점수:
- 관련성: {relevance:.1f}점
- 다양성: {diversity:.1f}점  
- 완전성: {completeness:.1f}점
- 일관성: {coherence:.1f}점
- 전체: {overall_score:.1f}점

{reflection_feedback}
"""
            except (ValueError, TypeError) as e:
                print(f"⚠️ 평가 점수 변환 오류: {e}")
                evaluation_text = f"""
평가 점수 변환 중 오류가 발생했습니다.
원본 데이터: {evaluation_scores}

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

        # 추가 지시사항 포함
        enhanced_prompt = f"""{prompt}

---
시스템 분석 정보:
{system_analysis}
---

위의 칵테일 정보와 시스템 분석 정보를 바탕으로, 사용자에게 도움이 되는 상세한 설명과 함께 칵테일을 추천해주세요.
추천 이유, 맛의 특징, 상황별 추천 등을 포함하여 설명하고, 답변 마지막에 "📋 시스템 분석 정보" 섹션을 추가하여 위의 시스템 분석 정보를 사용자 친화적으로 포함해주세요."""

        print(f"🎯 최종 응답 생성 중... ({task_type})")
        
        # OpenAI API 호출
        response = openai_client.generate(enhanced_prompt, max_tokens=1500)
        
        return response
        
    except Exception as e:
        print(f"❌ 응답 생성 오류: {e}")
        # 기본 응답 생성
        cocktails_text = format_cocktails_for_response(cocktails)
        return f"""죄송합니다. 응답 생성 중 오류가 발생했습니다.

다음 칵테일들을 추천드립니다:

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
            
            print(f"🔍 초기 평가 점수 확인:")
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