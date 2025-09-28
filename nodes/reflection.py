from typing import Dict, Any, List
import sys
import os

# 상위 디렉토리의 모듈 임포트를 위한 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.openai_client import OpenAIClient
from prompts.reflection_prompt import REFLECTION_PROMPT_TEMPLATE

# OpenAI 클라이언트 초기화
openai_client = OpenAIClient()

def format_search_results(results: List[Dict[str, Any]]) -> str:
    """
    검색 결과를 평가용 텍스트로 포맷팅
    
    Args:
        results: 검색된 칵테일 리스트
        
    Returns:
        포맷된 텍스트
    """
    if not results:
        return "검색 결과가 없습니다."
    
    formatted_lines = []
    for i, cocktail in enumerate(results, 1):
        name = cocktail.get('name', 'Unknown')
        category = cocktail.get('category', 'N/A')
        glass_type = cocktail.get('glassType', 'N/A')
        alcoholic = cocktail.get('alcoholic', 'N/A')
        description = cocktail.get('description', 'N/A')
        
        # 재료 정보
        ingredients = cocktail.get('ingredients', [])
        recipe_ingredients = cocktail.get('recipe_ingredients', [])
        
        if recipe_ingredients:
            ingredients_text = ", ".join([f"{item.get('measure', '')} {item.get('ingredient', '')}" 
                                        for item in recipe_ingredients])
        else:
            ingredients_text = ", ".join(ingredients)
        
        formatted_lines.append(f"{i}. {name}")
        formatted_lines.append(f"   카테고리: {category} | 글라스: {glass_type} | 알코올: {alcoholic}")
        formatted_lines.append(f"   재료: {ingredients_text}")
        formatted_lines.append(f"   설명: {description}")
        formatted_lines.append("")  # 빈 줄
    
    return "\n".join(formatted_lines)

def evaluate_search_quality(user_query: str, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    검색 결과의 품질을 4가지 기준으로 평가
    
    Args:
        user_query: 사용자 질문
        search_results: 검색 결과 리스트
        
    Returns:
        평가 결과 딕셔너리
    """
    try:
        # 기존 reflection 프롬프트 사용
        formatted_results = format_search_results(search_results)
        
        # 프롬프트에 데이터 삽입
        prompt = REFLECTION_PROMPT_TEMPLATE.format(
            user_query=user_query,
            num_results=len(search_results),
            search_results=formatted_results
        )
        
        print(f"🤔 검색 품질 평가 중...")
        
        # OpenAI API 호출
        response = openai_client.generate(prompt, response_format="json")
        
        # JSON 파싱
        evaluation = openai_client.parse_json_response(response)
        
        # 기본값으로 검증
        if "error" in evaluation:
            print(f"⚠️ 평가 파싱 오류, 기본값 사용")
            evaluation = {
                "relevance": 50.0,
                "diversity": 50.0,
                "completeness": 50.0,
                "coherence": 50.0,
                "overall_score": 50.0,
                "feedback": "평가 중 오류가 발생했습니다.",
                "suggestions": ["검색 결과를 다시 확인해주세요."],
                "should_retry": True
            }
        
        # 점수 검증 및 변환
        scores = {}
        for key in ["relevance", "diversity", "completeness", "coherence"]:
            try:
                scores[key] = float(evaluation.get(key, 50))
                # 0-100 범위 확인
                scores[key] = max(0, min(100, scores[key]))
            except (ValueError, TypeError):
                scores[key] = 50.0
        
        # 전체 점수 계산
        overall_score = sum(scores.values()) / len(scores)
        
        # 재시도 여부 결정 (80점 미만 시 재시도)
        should_retry = overall_score < 80
        
        result = {
            "relevance": scores["relevance"],
            "diversity": scores["diversity"], 
            "completeness": scores["completeness"],
            "coherence": scores["coherence"],
            "overall_score": overall_score,
            "feedback": evaluation.get("feedback", "평가 완료"),
            "suggestions": evaluation.get("suggestions", []),
            "should_retry": should_retry
        }
        
        print(f"📊 평가 완료: {overall_score:.1f}점 (재시도: {'예' if should_retry else '아니오'})")
        
        return result
        
    except Exception as e:
        print(f"❌ 품질 평가 오류: {e}")
        return {
            "relevance": 50.0,
            "diversity": 50.0,
            "completeness": 50.0,
            "coherence": 50.0,
            "overall_score": 50.0,
            "feedback": f"평가 중 오류 발생: {str(e)}",
            "suggestions": ["시스템 오류로 재시도가 필요합니다."],
            "should_retry": True
        }

def reflection(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    검색 결과의 품질을 평가하고 필요시 재검색을 결정하는 리플렉션 노드
    
    최대 3회까지 반복하며, 각 라운드마다 top-k를 1씩 증가시킴
    
    Args:
        state: 파이프라인 상태 딕셔너리
        
    Returns:
        평가 결과와 재시도 여부가 포함된 상태 딕셔너리
    """
    # 필요한 정보 추출
    user_query = state.get("input_text_with_image", state.get("input_text", ""))
    search_results = state.get("search_results", [])
    iteration_count = state.get("iteration_count", 0)
    current_top_k = state.get("current_top_k", 3)
    
    print(f"\n🔄 Reflection (라운드 {iteration_count + 1}/3, Top-K: {current_top_k})")
    
    # 검색 결과가 없는 경우
    if not search_results:
        print("❌ 검색 결과가 없어서 평가할 수 없습니다.")
        state["score"] = 0.0
        state["evaluation_scores"] = {
            "relevance": 0.0,
            "diversity": 0.0,
            "completeness": 0.0,
            "coherence": 0.0
        }
        state["reflection_feedback"] = "검색 결과가 없습니다."
        return state
    
    # 품질 평가 실행
    evaluation = evaluate_search_quality(user_query, search_results)
    
    # 평가 결과 저장
    state["score"] = evaluation["overall_score"]
    state["evaluation_scores"] = {
        "relevance": evaluation["relevance"],
        "diversity": evaluation["diversity"],
        "completeness": evaluation["completeness"],
        "coherence": evaluation["coherence"]
    }
    state["reflection_feedback"] = evaluation["feedback"]
    
    # 최고 점수 결과 업데이트 (동점 시 더 높은 라운드 선택)
    current_score = evaluation["overall_score"]
    best_score = state.get("best_result", {}).get("score", 0)
    
    should_update = False
    update_reason = ""
    
    if "best_result" not in state:
        should_update = True
        update_reason = "첫 번째 결과"
    elif current_score > best_score:
        should_update = True
        update_reason = f"더 높은 점수 ({current_score:.1f} > {best_score:.1f})"
    elif current_score == best_score:
        # 동점일 때는 더 높은 라운드(더 많은 칵테일) 선택
        should_update = True
        update_reason = f"동점이므로 다양성 증가 (Round {iteration_count + 1}, Top-{current_top_k})"
    
    if should_update:
        state["best_result"] = {
            "score": evaluation["overall_score"],
            "results": search_results.copy(),
            "evaluation": evaluation,
            "top_k": current_top_k,
            "iteration": iteration_count + 1
        }
        print(f"🏆 최고 결과 업데이트: {evaluation['overall_score']:.1f}점 ({update_reason})")
    
    # 반복 횟수 증가
    state["iteration_count"] = iteration_count + 1
    
    # 초기 결과 저장 (Round 1 완료 시)
    if state["iteration_count"] == 1:
        print(f"💾 초기 결과 저장 (Round 1, {len(search_results)}개 칵테일)")
        state["initial_search_results"] = search_results.copy()
        # 초기 평가 정보도 저장
        state["initial_evaluation_scores"] = {
            "relevance": evaluation["relevance"],
            "diversity": evaluation["diversity"],
            "completeness": evaluation["completeness"],
            "coherence": evaluation["coherence"]
        }
        state["initial_score"] = evaluation["overall_score"]
        state["initial_feedback"] = evaluation["feedback"]
        
        # 디버그: 저장되는 초기 평가 점수 확인
        print(f"🔍 초기 평가 점수 저장 디버그:")
        print(f"   - evaluation (원본): {evaluation}")
        print(f"   - initial_evaluation_scores (저장): {state['initial_evaluation_scores']}")
        print(f"   - initial_score (저장): {state['initial_score']}")
        # 초기 답변은 Generator에서 생성됨
    
    # 재시도 결정 로직
    should_retry = False
    
    # 1. 점수가 80점 이상이면 즉시 완료
    if evaluation["overall_score"] >= 80:
        print(f"✅ 품질 기준 충족 ({evaluation['overall_score']:.1f}점 >= 80점)")
        should_retry = False
        
        # 현재 결과를 최종 결과로 저장 (80점 이상 달성)
        current_round = state["iteration_count"]
        print(f"🏆 품질 기준 달성: Round {current_round} ({evaluation['overall_score']:.1f}점, Top-K: {current_top_k})")
        print(f"💾 최종 결과 저장 (Round {current_round}, {len(search_results)}개 칵테일)")
        
        state["final_search_results"] = search_results.copy()
        state["final_best_score"] = evaluation['overall_score']
        state["final_best_round"] = current_round
        state["final_best_top_k"] = current_top_k
        state["reflection_feedback"] = f"품질 기준 달성: {evaluation['overall_score']:.1f}점 (Round {current_round})"
    # 2. 최대 반복 횟수 도달 시 종료
    elif state["iteration_count"] >= 3:
        print(f"⏸️ 최대 반복 횟수 도달 (3회), 최고 점수 결과 선택")
        should_retry = False
        
        # 최고 점수 결과를 최종 결과로 사용
        if "best_result" in state:
            best_result = state["best_result"]
            final_score = best_result["score"]
            best_round = best_result["iteration"]
            best_top_k = best_result["top_k"]
            
            print(f"🏆 최고 점수 선택: Round {best_round} ({final_score:.1f}점, Top-K: {best_top_k})")
            print(f"💾 최종 결과 저장 (Round {best_round}, {len(best_result['results'])}개 칵테일)")
            
            state["final_search_results"] = best_result["results"].copy()
            state["final_best_score"] = final_score
            state["final_best_round"] = best_round
            state["final_best_top_k"] = best_top_k
            
            # 최종 답변은 Generator에서 생성됨
            state["reflection_feedback"] = f"3회 반복 완료: 최고 점수 {final_score:.1f}점 (Round {best_round})"
        else:
            # 폴백: best_result가 없으면 현재 결과 사용
            print(f"⚠️ best_result 없음, 현재 라운드 결과 사용")
            print(f"💾 최종 결과 저장 (Round 3, {len(search_results)}개 칵테일)")
            state["final_search_results"] = search_results.copy()
            state["final_best_score"] = evaluation['overall_score']
            state["final_best_round"] = 3
            state["final_best_top_k"] = current_top_k
            state["reflection_feedback"] = f"3회 반복 완료: 최종 점수 {evaluation['overall_score']:.1f}점"
    # 3. 그렇지 않으면 재시도
    else:
        should_retry = True
        # top-k 증가
        state["current_top_k"] = current_top_k + 1
        print(f"🔄 재시도 결정: Top-K를 {current_top_k + 1}로 증가")
        print(f"📋 피드백: {evaluation['feedback']}")
        if evaluation['suggestions']:
            print(f"💡 개선 제안: {', '.join(evaluation['suggestions'])}")
    
    # 디버그 정보 업데이트
    if "debug_info" not in state:
        state["debug_info"] = {}
    
    if "reflection_history" not in state["debug_info"]:
        state["debug_info"]["reflection_history"] = []
    
    state["debug_info"]["reflection_history"].append({
        "iteration": iteration_count + 1,
        "top_k": current_top_k,
        "score": evaluation["overall_score"],
        "scores": evaluation,
        "should_retry": should_retry,
        "results_count": len(search_results)
    })
    
    # should_retry 상태는 pipeline의 조건부 엣지에서 사용됨
    state["should_retry"] = should_retry
    
    # 캐시 보존 확인 (디버깅)
    if "full_ranked_cocktails" in state:
        print(f"🔍 Reflection 종료 시 캐시 상태: {len(state.get('full_ranked_cocktails', []))}개")
    else:
        print(f"⚠️ Reflection 종료 시 full_ranked_cocktails 키 없음")
    
    return state