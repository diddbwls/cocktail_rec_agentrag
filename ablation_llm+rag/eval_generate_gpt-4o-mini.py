# #!/usr/bin/env python3
# """
# gpt-4o-mini 모델 평가용 답변 생성 스크립트 (Ablation: Simple RAG)
# 이 파일을 실행하면 output 폴더에 gpt-4o-mini모델로 cocktail_eval_data에 대해 생성한 답변 파일이 생성됨

# Ablation 조건: task 분류와 reflection은 사용하지만, 복잡한 graph retrieval 대신 단순 유사도 비교만 사용
# """
# import pandas as pd
# import json
# import os
# import sys
# import time
# from pathlib import Path
# from tqdm import tqdm
# from typing import Dict, Any

# # 상위 디렉토리 추가
# current_dir = Path(__file__).parent
# parent_dir = current_dir.parent
# sys.path.append(str(parent_dir))

# ####################################################
# # 모델 설정 
# import utils.config as config
# config.LLM_MODEL = "gpt-4o-mini"
# EVAL_DATA_COUNT = 5  # 평가할 데이터 개수 (전체: None, 일부: 숫자)
# ####################################################

# from nodes.user_question import describe_image
# from nodes.task_classifier import query_classification
# from nodes.reflection import reflection
# from nodes.generator import generator
# from simple_retrieval import SimpleRetrieval

# # 평가 데이터 path (부모 디렉토리 기준)
# eval_data_path = './data/cocktail_eval_data.csv'

# # 질문 이미지 path
# question_image_path = './data/image/'

# # 출력 파일 path
# output_file = f'./ablation_llm+rag/output/{config.LLM_MODEL}_ablation_simple_rag.csv'

# def load_eval_data():
#     """평가 데이터 로드"""
#     df = pd.read_csv(eval_data_path)
#     if EVAL_DATA_COUNT is None:
#         return df  # 전체 데이터
#     else:
#         return df.iloc[0:EVAL_DATA_COUNT]  # 지정된 개수만

# def run_simple_rag_pipeline(user_question: str) -> Dict[str, Any]:
#     """단순 RAG 파이프라인 실행 (ablation: 복잡한 graph retrieval 대신 단순 유사도 비교)"""
#     state = {
#         "input_text_with_image": user_question,
#         "debug_info": {"reflection_history": []},
#         "current_top_k": config.FINAL_TOP_K,
#         "iteration_count": 0
#     }
    
#     # 1. Task Classification
#     print("📊 Task Classification...")
#     state = query_classification(state)
    
#     retriever = SimpleRetrieval()
#     max_iterations = 3
#     best_result = None
    
#     # Iterative retrieval with reflection (like original, but with simple similarity)
#     for iteration in range(max_iterations):
#         current_top_k = state["current_top_k"]
#         print(f"\n🔄 Round {iteration + 1}: Top-K = {current_top_k}")
        
#         # 2. Simple Retrieval (similarity-based)
#         print("🔍 Simple Retrieval...")
#         search_results = retriever.retrieve(user_question, top_k=current_top_k)
#         state["search_results"] = search_results
        
#         # Store initial results
#         if iteration == 0:
#             state["initial_search_results"] = search_results
        
#         # 3. Reflection
#         print("🤔 Reflection...")
#         state = reflection(state)
        
#         # Update best result if better
#         current_score = state.get("score", 0)
#         if best_result is None or current_score > best_result["score"]:
#             best_result = {
#                 "score": current_score,
#                 "results": search_results.copy(),
#                 "iteration": iteration + 1,
#                 "top_k": current_top_k
#             }
        
#         # Check if should continue
#         should_retry = state.get("should_retry", False)
#         if not should_retry or current_score >= 80:
#             print(f"✅ Stopping: should_retry={should_retry}, score={current_score}")
#             break
            
#         # Increase top_k for next iteration
#         state["current_top_k"] = current_top_k + 1
#         state["iteration_count"] = iteration + 1
    
#     # Use best result
#     final_search_results = best_result["results"] if best_result else search_results
#     state["final_search_results"] = final_search_results
#     state["search_results"] = final_search_results  # For generator
    
#     # 4. Response Generation
#     print("📤 Response Generation...")
#     state = generator(state)
    
#     # Extract data for ablation
#     final_response = state.get("final_response", "")
#     final_context = retriever.format_cocktails_as_context(final_search_results)
    
#     # Get reflection scores from evaluation_scores
#     eval_scores = state.get("evaluation_scores", {})
#     reflection_result = {
#         "relevance": eval_scores.get("relevance", 0),
#         "diversity": eval_scores.get("diversity", 0),
#         "completeness": eval_scores.get("completeness", 0),
#         "coherence": eval_scores.get("coherence", 0),
#         "overall_score": state.get("score", 0),
#         "should_retry": state.get("should_retry", False)
#     }
    
#     # Compile final state  
#     final_state = {
#         "input_text_with_image": user_question,
#         "task_type": state.get("task_type", "Unknown"),
#         "task_confidence": state.get("task_confidence", 0.0),
#         "initial_search_results": state.get("initial_search_results", []),
#         "initial_response": state.get("initial_response", ""),
#         "final_cocktails": final_search_results,
#         "final_response": final_response,
#         "final_context": final_context,
#         "iteration_count": state.get("iteration_count", 0) + 1,
#         "reflection_result": reflection_result,
#         "debug_info": state.get("debug_info", {})
#     }
    
#     return final_state

# def extract_pipeline_data(state: Dict[str, Any]) -> Dict[str, Any]:
#     """파이프라인 state에서 평가 데이터 추출"""
#     try:
#         # 기본 정보
#         user_question = state.get("input_text_with_image", "")
#         final_cocktails = state.get("final_cocktails", [])
        
#         # Round 1 정보 (initial)
#         initial_search_results = state.get("initial_search_results", [])
#         round1_cocktails = [c.get('name', '') for c in initial_search_results]
        
#         # 답변 정보
#         initial_response = state.get("initial_response", "")
#         final_response = state.get("final_response", "")
        
#         # 컨텍스트 정보
#         round1_context = state.get("final_context", "")
#         final_context = state.get("final_context", "")
        
#         # reflection 결과
#         reflection_result = state.get("reflection_result", {})
        
#         # scores 딕셔너리를 개별 컬럼으로 분리하는 함수
#         def extract_scores(scores_dict, prefix):
#             """scores 딕셔너리를 개별 컬럼으로 분리"""
#             return {
#                 f'{prefix}_relevance': scores_dict.get('relevance', 0),
#                 f'{prefix}_diversity': scores_dict.get('diversity', 0),
#                 f'{prefix}_completeness': scores_dict.get('completeness', 0),
#                 f'{prefix}_coherence': scores_dict.get('coherence', 0),
#                 f'{prefix}_overall_score': scores_dict.get('overall_score', 0),
#                 f'{prefix}_should_retry': scores_dict.get('should_retry', False)
#             }
        
#         # Round1과 Final scores를 개별 컬럼으로 분리
#         round1_score_columns = extract_scores(reflection_result, 'round1')
#         final_score_columns = extract_scores(reflection_result, 'final')
        
#         result = {
#             'user_question': user_question,
#             'task_type': state.get("task_type", "Unknown"),
#             'task_confidence': state.get("task_confidence", 0.0),
#             'round1_cocktails': round1_cocktails,
#             'round1_context': round1_context,
#             'round1_answer': initial_response,
#             'final_cocktails': [c.get('name', '') for c in final_cocktails],
#             'final_context': final_context,
#             'final_answer': final_response,
#             'iteration_count': state.get("iteration_count", 1),
#             'final_round': 1
#         }
        
#         # scores 개별 컬럼들 추가
#         result.update(round1_score_columns)
#         result.update(final_score_columns)
        
#         return result
        
#     except Exception as e:
#         print(f"⚠️ 데이터 추출 오류: {e}")
#         error_result = {
#             'user_question': state.get("input_text_with_image", ""),
#             'task_type': "Unknown",
#             'task_confidence': 0.0,
#             'round1_cocktails': [],
#             'round1_context': "",
#             'round1_answer': "",
#             'final_cocktails': [],
#             'final_context': "",
#             'final_answer': "",
#             'iteration_count': 0,
#             'final_round': 0,
#             # Round1 scores (기본값)
#             'round1_relevance': 0,
#             'round1_diversity': 0,
#             'round1_completeness': 0,
#             'round1_coherence': 0,
#             'round1_overall_score': 0,
#             'round1_should_retry': False,
#             # Final scores (기본값)
#             'final_relevance': 0,
#             'final_diversity': 0,
#             'final_completeness': 0,
#             'final_coherence': 0,
#             'final_overall_score': 0,
#             'final_should_retry': False
#         }
#         return error_result

# def run_evaluation(test_case):
#     """단일 테스트 케이스 평가 실행 (Ablation: Simple RAG)"""
#     print(f"\n🔍 평가 시작: Index {test_case['Index']}")
    
#     try:
#         # 이미지 경로 구성
#         image_path = os.path.join(question_image_path, f"{test_case['fileNumber']}.jpeg")
#         print(f"📸 이미지: {image_path}")
        
#         # 이미지 설명 생성
#         print("🖼️ 이미지 설명 생성 중...")
#         if os.path.exists(image_path):
#             image_description = describe_image(image_path)
#             print(f"✅ 이미지 설명: {image_description}")
#         else:
#             print(f"⚠️ 이미지 파일 없음: {image_path}")
#             image_description = f"[이미지 설명 생성 실패: {image_path}]"
        
#         # 사용자 질문 조합
#         user_question = f"{image_description} {test_case['query_EN']}"
#         print(f"❓ 사용자 질문: {user_question}")
        
#         # 단순 RAG 파이프라인 실행
#         print("🚀 Simple RAG 파이프라인 실행 중...")
#         result = run_simple_rag_pipeline(user_question)
        
#         # 평가 데이터 추출
#         print("📊 평가 데이터 추출 중...")
#         pipeline_data = extract_pipeline_data(result)
        
#         # 기본 정보와 결합
#         evaluation_result = {
#             # 기존 컬럼들
#             'Task': test_case['Task'],
#             'Index': test_case['Index'],
#             'image': test_case['image'],
#             'query_KO': test_case['query_KO'],
#             'query_EN': test_case['query_EN'],
#             'fileNumber': test_case['fileNumber'],
#             'imageCocktailName': test_case['imageCocktailName'],
            
#             # RAGAS 평가용 핵심 데이터
#             'question': pipeline_data['user_question'],
#             'task_type': pipeline_data['task_type'],
#             'task_confidence': pipeline_data['task_confidence'],
#             'contexts': json.dumps([pipeline_data['final_context']], ensure_ascii=False),
#             'answer': pipeline_data['final_answer'],
            
#             # 추가 상세 데이터
#             'round1_cocktails': json.dumps(pipeline_data['round1_cocktails'], ensure_ascii=False),
#             'round1_context': pipeline_data['round1_context'],
#             'round1_answer': pipeline_data['round1_answer'],
#             'final_cocktails': json.dumps(pipeline_data['final_cocktails'], ensure_ascii=False),
#             'final_context': pipeline_data['final_context'],
#             'final_answer': pipeline_data['final_answer'],
#             'iteration_count': pipeline_data['iteration_count'],
#             'final_round': pipeline_data['final_round'],
            
#             # Round1 scores (개별 컬럼)
#             'round1_relevance': pipeline_data['round1_relevance'],
#             'round1_diversity': pipeline_data['round1_diversity'],
#             'round1_completeness': pipeline_data['round1_completeness'],
#             'round1_coherence': pipeline_data['round1_coherence'],
#             'round1_overall_score': pipeline_data['round1_overall_score'],
#             'round1_should_retry': pipeline_data['round1_should_retry'],
            
#             # Final scores (개별 컬럼)
#             'final_relevance': pipeline_data['final_relevance'],
#             'final_diversity': pipeline_data['final_diversity'],
#             'final_completeness': pipeline_data['final_completeness'],
#             'final_coherence': pipeline_data['final_coherence'],
#             'final_overall_score': pipeline_data['final_overall_score'],
#             'final_should_retry': pipeline_data['final_should_retry'],
            
#             # Ablation 정보
#             'method': 'simple_rag'
#         }
        
#         print(f"✅ 평가 완료: Index {test_case['Index']}")
#         return evaluation_result
        
#     except Exception as e:
#         print(f"❌ 평가 오류 (Index {test_case['Index']}): {e}")
#         import traceback
#         traceback.print_exc()
#         return None

# def main():
#     """메인 실행 함수 (Ablation: Simple RAG)"""
#     start_time = time.time()
    
#     print(f"🚀 {config.LLM_MODEL} 모델 Ablation 평가 시작 (Simple RAG)")
#     print(f"📋 사용 모델: {config.LLM_MODEL}")
#     print(f"🧪 Ablation 조건: task 분류 + reflection + 단순 유사도 검색")
    
#     # 평가 데이터 로드
#     print("\n📂 평가 데이터 로드 중...")
#     eval_data = load_eval_data()
#     print(f"✅ 테스트 케이스 수: {len(eval_data)}")
    
#     # 출력 디렉토리 생성
#     os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
#     # 각 테스트 케이스 실행 (tqdm 진행률 표시)
#     results = []
#     for _, test_case in tqdm(eval_data.iterrows(), total=len(eval_data), desc="Simple RAG 평가용 답변 생성"):
#         result = run_evaluation(test_case)
#         if result:
#             results.append(result)
    
#     # 결과 저장
#     if results:
#         print(f"\n💾 결과 저장 중... ({len(results)}개 케이스)")
        
#         df_results = pd.DataFrame(results)
#         df_results.to_csv(output_file, index=False, encoding='utf-8-sig')
        
#         print(f"✅ 저장 완료: {output_file}")
#         print(f"📊 컬럼 수: {len(df_results.columns)}")
#         print(f"📈 데이터 행 수: {len(df_results)}")
        
#         # 컬럼 정보 출력
#         print("\n📋 생성된 컬럼들:")
#         for i, col in enumerate(df_results.columns, 1):
#             print(f"  {i:2d}. {col}")
#     else:
#         print("❌ 생성된 결과가 없습니다.")
    
#     # 전체 실행시간 출력
#     end_time = time.time()
#     total_time = end_time - start_time
#     minutes = int(total_time // 60)
#     seconds = total_time % 60
    
#     print(f"\n⏱️ 전체 실행시간: {minutes}분 {seconds:.1f}초 ({total_time:.1f}초)")
#     if len(results) > 0:
#         avg_time = total_time / len(results)
#         print(f"📈 평균 케이스당 처리시간: {avg_time:.1f}초")

# if __name__ == "__main__":
#     main()