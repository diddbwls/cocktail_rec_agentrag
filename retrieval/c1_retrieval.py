from typing import List, Dict, Any
import sys
import os

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.config import get_config, get_c1_config
from retrieval.base_retrieval import BaseRetrieval

class C1Retrieval(BaseRetrieval):
    """C1 태스크: 색상-재료 매칭 기반 시각 칵테일 추천 검색"""
    
    def __init__(self, use_python_config: bool = True):
        """Initialize C1 retrieval system"""
        if use_python_config:
            # Python 설정 사용
            config = get_config()
            c1_config = get_c1_config()
        else:
            # 기존 JSON 설정 사용 (하위 호환성)
            import json
            with open("config.json", 'r') as f:
                config = json.load(f)
            c1_config = config['c1_config']
        
        # 기본 클래스 초기화
        super().__init__(config, c1_config)
        self.c1_config = c1_config  # 편의를 위해 별도 저장
    
    def extract_keywords(self, user_question: str) -> Dict[str, List[str]]:
        """LLM을 사용하여 사용자 질문에서 키워드 추출 및 노드별 분류"""
        result = super().extract_keywords(user_question)
        
        # C1 특화: 하위 호환성을 위해 기존 ingredient 키도 처리
        if 'ingredient' in result and 'include_ingredients' not in result:
            result['include_ingredients'] = result['ingredient']
        if 'exclude_ingredients' not in result:
            result['exclude_ingredients'] = []
        if 'visual_keywords' not in result:
            result['visual_keywords'] = []
            
        return result

    def get_cocktail_details(self, cocktail_names: List[str]) -> List[Dict[str, Any]]:
        """칵테일 상세 정보 가져오기"""
        cocktails = []
        with self.driver.session() as session:
            for name in cocktail_names:
                query = """
                MATCH (c:Cocktail {name: $name})
                OPTIONAL MATCH (c)-[:CATEGORY]->(cat:Category)
                OPTIONAL MATCH (c)-[:HAS_GLASSTYPE]->(g:GlassType)
                OPTIONAL MATCH (c)-[:HAS_INGREDIENT]->(i:Ingredient)
                RETURN c.name as name, 
                       c.description as description,
                       c.instructions as instructions,
                       c.imageDescription as imageDescription,
                       c.alcoholic as alcoholic,
                       cat.name as category,
                       g.name as glassType,
                       collect(DISTINCT i.name) as ingredients
                """
                
                result = session.run(query, {'name': name})
                record = result.single()
                
                if record:
                    cocktails.append({
                        'name': record['name'],
                        'description': record['description'],
                        'instructions': record['instructions'],
                        'imageDescription': record['imageDescription'],
                        'alcoholic': record['alcoholic'],
                        'category': record['category'],
                        'glassType': record['glassType'],
                        'ingredients': record['ingredients']
                    })
        
        return cocktails
    
    
    
    def search_cocktails_by_question_embedding(self, user_question: str, top_k: int = None) -> List[str]:
        """질문 전체를 임베딩하여 imageDescription_embedding과 유사도 검색"""
        question_embedding = self.get_embedding(user_question)
        
        if top_k is None:
            top_k = self.c1_config['initial_top_k']
        
        cocktails = []
        with self.driver.session() as session:
            # Vector similarity search for imageDescription using question embedding
            query = """
            CALL db.index.vector.queryNodes('cocktail_imageDescription_embedding', $k, $embedding)
            YIELD node, score
            WHERE score >= $threshold
            RETURN node.name as name, score
            ORDER BY score DESC
            """
            
            result = session.run(query, {
                'k': top_k,
                'embedding': question_embedding,
                'threshold': self.c1_config['similarity_threshold']
            })
            
            for record in result:
                cocktails.append(record['name'])
        
        return cocktails

    def find_best_colored_ingredient(self, cocktail_names: List[str], color_keyword: str) -> str:
        """모든 초기 칵테일의 재료 중에서 색상과 가장 유사한 재료 1개 선정"""
        all_ingredients = []
        
        with self.driver.session() as session:
            # 모든 초기 칵테일의 재료 수집
            for cocktail_name in cocktail_names:
                query = """
                MATCH (c:Cocktail {name: $name})-[:HAS_INGREDIENT]->(i:Ingredient)
                RETURN i.name as ingredient
                """
                
                result = session.run(query, {'name': cocktail_name})
                for record in result:
                    if record['ingredient'] not in all_ingredients:
                        all_ingredients.append(record['ingredient'])
            
            if not all_ingredients:
                return ""
            
            # 색상 키워드와 모든 재료의 유사도 계산
            color_embedding = self.get_embedding(color_keyword)
            ingredient_similarities = []
            
            for ingredient in all_ingredients:
                ingredient_embedding = self.get_embedding(ingredient)
                # 코사인 유사도 계산
                similarity = self.calculate_cosine_similarity(color_embedding, ingredient_embedding)
                ingredient_similarities.append((ingredient, similarity))
            
            # 가장 유사도 높은 재료 1개 선정
            ingredient_similarities.sort(key=lambda x: x[1], reverse=True)
            best_ingredient, best_similarity = ingredient_similarities[0]
            
            print(f"   → {color_keyword}와 가장 유사한 재료: {best_ingredient} (유사도: {best_similarity:.3f})")
            return best_ingredient

    def expand_by_ingredient_sharing(self, best_ingredient: str, initial_cocktails: List[str], user_question: str) -> List[str]:
        """재료 공유 관계를 활용해 imageDescription 유사도 높은 2-3개 칵테일 선정"""
        if not best_ingredient:
            return []
            
        question_embedding = self.get_embedding(user_question)
        
        with self.driver.session() as session:
            # 해당 재료를 가진 칵테일들 검색 (초기 칵테일 제외)
            query = """
            MATCH (c:Cocktail)-[:HAS_INGREDIENT]->(i:Ingredient {name: $ingredient})
            WHERE NOT c.name IN $exclude_cocktails
            AND c.imageDescription_embedding IS NOT NULL
            RETURN c.name as name, c.imageDescription_embedding as embedding
            """
            
            result = session.run(query, {
                'ingredient': best_ingredient,
                'exclude_cocktails': initial_cocktails
            })
            
            # 각 칵테일과 사용자 질문의 imageDescription 유사도 계산
            cocktail_similarities = []
            for record in result:
                if record['embedding']:
                    cocktail_embedding = record['embedding']
                    similarity = self.calculate_cosine_similarity(question_embedding, cocktail_embedding)
                    cocktail_similarities.append((record['name'], similarity))
            
            # 유사도 높은 순으로 정렬하여 상위 2-3개 선택
            if cocktail_similarities:
                cocktail_similarities.sort(key=lambda x: x[1], reverse=True)
                # 최대 3개까지 선택 (단, 유사도 0.3 이상만)
                selected_cocktails = []
                for name, similarity in cocktail_similarities[:3]:
                    if similarity > 0.3:  # 최소 유사도 임계값
                        selected_cocktails.append(name)
                        print(f"   → {best_ingredient} 재료로 확장: {name} (이미지 유사도: {similarity:.3f})")
                
                print(f"   → 색상 재료로 확장된 칵테일: {len(selected_cocktails)}개")
                return selected_cocktails
        
        return []

    def rank_all_candidates_by_similarity(self, user_question: str, cocktail_names: List[str]) -> List[str]:
        """전체 후보를 유사도 높은순→낮은순으로 정렬하여 이름 리스트 반환"""
        if not cocktail_names:
            return []
        
        question_embedding = self.get_embedding(user_question)
        cocktail_similarities = []
        
        with self.driver.session() as session:
            for name in cocktail_names:
                # 각 칵테일의 imageDescription_embedding 가져오기
                query = """
                MATCH (c:Cocktail {name: $name})
                RETURN c.imageDescription_embedding as embedding
                """
                
                result = session.run(query, {'name': name})
                record = result.single()
                
                if record and record['embedding']:
                    # 코사인 유사도 계산
                    cocktail_embedding = record['embedding']
                    similarity = self.calculate_cosine_similarity(question_embedding, cocktail_embedding)
                    cocktail_similarities.append((name, similarity))
                else:
                    # embedding이 없으면 낮은 점수로 설정
                    cocktail_similarities.append((name, 0.0))
        
        # 유사도 기준으로 정렬 (높은순→낮은순)
        cocktail_similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 전체 유사도 랭킹 출력
        print(f"전체 후보 유사도 랭킹:")
        for i, (name, similarity) in enumerate(cocktail_similarities, 1):
            print(f"   {i}. {name} (유사도: {similarity:.3f})")
        
        # 유사도순 정렬된 이름 리스트 반환
        return [name for name, _ in cocktail_similarities]

    def retrieve(self, user_question: str) -> Dict[str, Any]:
        """색상 기반 재료 매칭을 활용한 시각 검색 알고리즘"""
        print(f"C1 Retrieval (색상-재료 기반): 사용자 질문 - {user_question}")
        
        # 1단계: 키워드 추출 (visual_keywords 포함)
        keywords = self.extract_keywords(user_question)
        print(f"1단계 - 키워드 추출: {keywords}")
        visual_keywords = keywords.get('visual_keywords', [])
        
        # 2단계: 초기 시각 검색 - 질문과 imageDescription 유사도로 top-k 선정
        initial_candidates = self.search_cocktails_by_question_embedding(user_question)
        print(f"2단계 - 초기 시각 검색: {len(initial_candidates)}개 칵테일 선정")
        print(f"   → 선정된 칵테일: {initial_candidates}")
        
        if not initial_candidates:
            print("❌ 초기 후보를 찾을 수 없습니다.")
            return {'results': [], 'full_ranked_names': [], 'current_top_k': 0}
        
        # 3단계: 색상 키워드가 있으면 색상-재료 매칭으로 확장
        expanded_cocktails = []
        if visual_keywords:
            print(f"3단계 - 색상 기반 재료 매칭 (색상: {visual_keywords})")
            
            for color in visual_keywords:
                # 모든 초기 칵테일의 재료 중에서 색상과 가장 유사한 재료 1개 선정
                best_ingredient = self.find_best_colored_ingredient(initial_candidates, color)
                
                if best_ingredient:
                    # 해당 재료를 가진 칵테일 중에서 imageDescription 유사도 높은 2-3개 확장
                    ingredient_cocktails = self.expand_by_ingredient_sharing(best_ingredient, initial_candidates, user_question)
                    if ingredient_cocktails:
                        expanded_cocktails.extend(ingredient_cocktails)
            
            print(f"   → 총 확장된 칵테일: {len(expanded_cocktails)}개")
        else:
            print("3단계 - 색상 키워드 없음, 확장 검색 생략")
        
        # 초기 선정 + 확장 칵테일 합치기 (중복 제거)
        all_candidates = list(set(initial_candidates + expanded_cocktails))
        print(f"\n전체 후보 (중복 제거): {len(all_candidates)}개 칵테일")
        if len(all_candidates) <= 10:
            print(f"   → 전체 후보: {all_candidates}")
        else:
            print(f"   → 전체 후보 (처음 10개): {all_candidates[:10]}...")
        
        # 4단계: 전체 후보를 유사도순으로 정렬
        similarity_ranked_names = self.rank_all_candidates_by_similarity(user_question, all_candidates)
        
        # 현재 라운드에 필요한 만큼만 선택
        final_top_k = self.config.get('final_top_k', self.c1_config['final_top_k'])
        print(f"🔧 최종 선정 개수: {final_top_k}개 (config: {self.config.get('final_top_k')}, c1_config: {self.c1_config['final_top_k']})")
        current_round_names = similarity_ranked_names[:final_top_k]
        
        # 상세 정보 가져오기
        current_results = self.get_cocktail_details(current_round_names)
        print(f"4단계 - C1 검색 완료: {len(current_results)}개 결과 (전체 랭킹: {len(similarity_ranked_names)}개)")
        
        # dict 형태로 반환 (캐싱을 위해)
        return {
            'results': current_results,
            'full_ranked_names': similarity_ranked_names,  # 전체 유사도순 이름 리스트
            'current_top_k': final_top_k
        }
