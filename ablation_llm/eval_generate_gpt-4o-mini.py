# #!/usr/bin/env python3
# """
# gpt-4o-mini 모델 평가용 답변 생성 스크립트 (Ablation: Direct LLM)
# 이 파일을 실행하면 output 폴더에 gpt-4o-mini모델로 cocktail_eval_data에 대해 생성한 답변 파일이 생성됨

# Ablation 조건: 복잡한 메커니즘 없이 이미지 설명 + 텍스트 질문을 LLM에 직접 전달하여 답변 생성
# """
# import pandas as pd
# import json
# import os
# import sys
# import time
# from pathlib import Path
# from tqdm import tqdm

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
# from utils.llm_model import get_llm

# # 평가 데이터 path (부모 디렉토리 기준)
# eval_data_path = './data/cocktail_eval_data.csv'

# # 질문 이미지 path
# question_image_path = './data/image/'

# # 출력 파일 path
# output_file = f'./ablation_llm/output/{config.LLM_MODEL}_ablation_llm.csv'

# def load_eval_data():
#     """평가 데이터 로드"""
#     df = pd.read_csv(eval_data_path)
#     if EVAL_DATA_COUNT is None:
#         return df  # 전체 데이터
#     else:
#         return df.iloc[0:EVAL_DATA_COUNT]  # 지정된 개수만

# def generate_direct_answer(user_question: str) -> str:
#     """LLM에 직접 질문하여 답변 생성 (ablation: 복잡한 메커니즘 없음)"""
#     llm = get_llm(config.LLM_MODEL)
    
#     prompt = f"""Based on the following question about cocktails, provide a helpful recommendation.

# User Question: {user_question}

# Please provide a detailed cocktail recommendation that addresses the user's request. Include specific cocktail names, ingredients, and explanations for your recommendations."""
    
#     try:
#         answer = llm.generate(prompt=prompt, temperature=config.TEMPERATURE)
#         return answer
#     except Exception as e:
#         print(f"⚠️ LLM 답변 생성 오류: {e}")
#         return "답변 생성에 실패했습니다."

# def run_evaluation(test_case):
#     """단일 테스트 케이스 평가 실행 (Ablation: Direct LLM)"""
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
        
#         # 직접 LLM 답변 생성 (ablation: 복잡한 메커니즘 제거)
#         print("🤖 LLM 직접 답변 생성 중...")
#         answer = generate_direct_answer(user_question)
        
#         # 평가 결과 구성 (단순화된 형태)
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
#             'question': user_question,
#             'contexts': json.dumps([""], ensure_ascii=False),  # 컨텍스트 없음 (ablation)
#             'answer': answer,
            
#             # Ablation 정보
#             'method': 'direct_llm',
#             'task_type': 'N/A',
#             'task_confidence': 0.0,
#             'iteration_count': 0
#         }
        
#         print(f"✅ 평가 완료: Index {test_case['Index']}")
#         return evaluation_result
        
#     except Exception as e:
#         print(f"❌ 평가 오류 (Index {test_case['Index']}): {e}")
#         import traceback
#         traceback.print_exc()
#         return None

# def main():
#     """메인 실행 함수 (Ablation: Direct LLM)"""
#     start_time = time.time()
    
#     print(f"🚀 {config.LLM_MODEL} 모델 Ablation 평가 시작 (Direct LLM)")
#     print(f"📋 사용 모델: {config.LLM_MODEL}")
#     print(f"🧪 Ablation 조건: 복잡한 메커니즘 없이 LLM 직접 답변")
    
#     # 평가 데이터 로드
#     print("\n📂 평가 데이터 로드 중...")
#     eval_data = load_eval_data()
#     print(f"✅ 테스트 케이스 수: {len(eval_data)}")
    
#     # 출력 디렉토리 생성
#     os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
#     # 각 테스트 케이스 실행 (tqdm 진행률 표시)
#     results = []
#     for _, test_case in tqdm(eval_data.iterrows(), total=len(eval_data), desc="Ablation 평가용 답변 생성"):
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