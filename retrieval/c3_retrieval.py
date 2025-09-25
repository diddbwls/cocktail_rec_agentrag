from typing import List, Dict, Any
import sys
import os

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.config import get_config, get_c3_config
from retrieval.base_retrieval import BaseRetrieval
import warnings

# Disable Neo4j logging for cleaner output
import logging
logging.getLogger("neo4j").setLevel(logging.WARNING)

class C3Retrieval(BaseRetrieval):
    """C3 태스크: Multi-hop 재료 확장 검색 기반 칵테일 추천"""
    
    def __init__(self, use_python_config: bool = True):
        """Initialize C3 retrieval system"""
        if use_python_config:
            # Python 설정 사용
            config = get_config()
            c3_config = get_c3_config()
        else:
            # 기존 JSON 설정 사용 (하위 호환성)
            import json
            with open("config.json", 'r') as f:
                config = json.load(f)
            c3_config = config['c3_config']
        
        # 기본 클래스 초기화
        super().__init__(config, c3_config)
        self.c3_config = c3_config  # 편의를 위해 별도 저장
    
    def extract_ingredients_and_cocktails(self, user_question: str) -> Dict[str, List[str]]:
        """LLM을 사용하여 사용자 질문에서 재료와 칵테일 이름 추출"""
        result = super().extract_keywords(user_question)
        
        # C3 특화: 필수 키만 확인
        if 'ingredients' not in result:
            result['ingredients'] = []
        if 'cocktail_names' not in result:
            result['cocktail_names'] = []
                
        return result
    
    def multi_hop_ingredient_expansion(self, ingredients: List[str]) -> List[str]:
        """Multi-hop 재료 확장 검색: 재료 → 칵테일 → 공통재료 → 새로운 칵테일"""
        if not ingredients:
            return []
        
        with self.driver.session(notifications_disabled_categories=["UNRECOGNIZED", "PERFORMANCE", "DEPRECATION", "HINT"]) as session:
            # 3-hop 검색을 단일 쿼리로 처리
            query = """
            // 1-hop: 사용자 재료들을 가진 칵테일들 찾기
            MATCH (c1:Cocktail)-[:HAS_INGREDIENT]->(i1:Ingredient)
            WHERE i1.name IN $ingredients
            WITH c1, count(i1) as matched_count
            WHERE matched_count >= $min_match
            
            // 2-hop: 그 칵테일들이 공통으로 사용하는 다른 재료들 발견
            MATCH (c1)-[:HAS_INGREDIENT]->(i2:Ingredient)
            WHERE NOT i2.name IN $ingredients
            WITH i2, count(DISTINCT c1) as cocktail_usage_count
            WHERE cocktail_usage_count >= $min_usage  // 최소 N개 칵테일에서 사용되는 재료만
            
            // 3-hop: 그 재료들을 사용하는 새로운 칵테일들 탐색
            MATCH (c2:Cocktail)-[:HAS_INGREDIENT]->(i2)
            WITH c2, count(DISTINCT i2) as expansion_strength
            
            // 원래 재료도 일부 가지고 있으면 더 좋음
            OPTIONAL MATCH (c2)-[:HAS_INGREDIENT]->(i_orig:Ingredient)
            WHERE i_orig.name IN $ingredients
            WITH c2, expansion_strength, count(i_orig) as original_ingredient_bonus
            
            RETURN DISTINCT c2.name as name, 
                   expansion_strength, 
                   original_ingredient_bonus,
                   (expansion_strength + original_ingredient_bonus) as total_strength
            ORDER BY total_strength DESC, expansion_strength DESC
            LIMIT $top_k
            """
            
            result = session.run(query, {
                'ingredients': ingredients,
                'min_match': self.c3_config['min_ingredient_match'],
                'min_usage': self.c3_config['min_cocktail_usage'],
                'top_k': self.c3_config['expansion_top_k']
            })
            
            expanded_cocktails = []
            for record in result:
                expanded_cocktails.append(record['name'])
                
            print(f"   → Multi-hop 확장 결과: {len(expanded_cocktails)}개 칵테일")
            if expanded_cocktails:
                print(f"      상위 5개: {expanded_cocktails[:5]}")
            
            return expanded_cocktails
    
    def find_cocktails_by_name_similarity(self, cocktail_names: List[str]) -> List[str]:
        """칵테일 이름 유사도로 직접 검색"""
        if not cocktail_names:
            return []
        
        found_cocktails = []
        with self.driver.session(notifications_disabled_categories=["UNRECOGNIZED", "PERFORMANCE", "DEPRECATION", "HINT"]) as session:
            for name in cocktail_names:
                # 정확한 이름 매치 먼저
                query = """
                MATCH (c:Cocktail)
                WHERE toLower(c.name) CONTAINS toLower($name)
                RETURN c.name as name
                LIMIT 3
                """
                
                result = session.run(query, {'name': name})
                for record in result:
                    if record['name'] not in found_cocktails:
                        found_cocktails.append(record['name'])
                
                # 임베딩 기반 유사 검색
                if len(found_cocktails) < 3:
                    name_embedding = self.get_embedding(name)
                    similar_cocktails = self.find_similar_cocktails_by_embedding(name_embedding, session, top_k=3)
                    for cocktail_name, similarity in similar_cocktails:
                        if cocktail_name not in found_cocktails and similarity > self.c3_config['name_similarity_threshold']:
                            found_cocktails.append(cocktail_name)
        
        print(f"   → 이름 유사도 검색: {len(found_cocktails)}개 칵테일")
        return found_cocktails[:self.c3_config['initial_top_k']]
    
    def find_initial_cocktails_by_ingredients(self, ingredients: List[str]) -> List[str]:
        """재료로 초기 칵테일 후보 찾기 (정확한 매치 우선)"""
        if not ingredients:
            return []
        
        with self.driver.session(notifications_disabled_categories=["UNRECOGNIZED", "PERFORMANCE", "DEPRECATION", "HINT"]) as session:
            # 정확한 재료 매치로 칵테일 찾기
            query = """
            MATCH (c:Cocktail)-[:HAS_INGREDIENT]->(i:Ingredient)
            WHERE i.name IN $ingredients
            WITH c, count(i) as matched_ingredients
            RETURN c.name as name, matched_ingredients
            ORDER BY matched_ingredients DESC
            LIMIT $top_k
            """
            
            result = session.run(query, {
                'ingredients': ingredients,
                'top_k': self.c3_config['initial_top_k']
            })
            
            initial_cocktails = []
            for record in result:
                initial_cocktails.append(record['name'])
            
            print(f"   → 재료 기반 초기 검색: {len(initial_cocktails)}개 칵테일")
            return initial_cocktails
    
    def find_similar_cocktails_by_embedding(self, embedding: List[float], session, top_k: int = 5):
        """임베딩 기반 칵테일 유사도 검색"""
        query = """
        MATCH (c:Cocktail)
        WHERE c.name_embedding IS NOT NULL
        RETURN c.name as name, c.name_embedding as embedding
        """
        
        result = session.run(query)
        similarities = []
        
        for record in result:
            if record['embedding']:
                cocktail_embedding = record['embedding']
                similarity = self.calculate_cosine_similarity(embedding, cocktail_embedding)
                similarities.append((record['name'], similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def remove_duplicates_preserve_order(self, cocktail_list: List[str]) -> List[str]:
        """중복 제거하면서 순서 보존"""
        seen = set()
        result = []
        for item in cocktail_list:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result
    
    def rank_by_image_similarity(self, user_question: str, cocktail_names: List[str]) -> List[str]:
        """사용자 질문과 칵테일들의 imageDescription 임베딩 유사도로 순위 매기기"""
        if not cocktail_names:
            return []
        
        question_embedding = self.get_embedding(user_question)
        cocktail_similarities = []
        
        with self.driver.session(notifications_disabled_categories=["UNRECOGNIZED", "PERFORMANCE", "DEPRECATION", "HINT"]) as session:
            for name in cocktail_names:
                query = """
                MATCH (c:Cocktail {name: $name})
                RETURN c.imageDescription_embedding as embedding, c.imageDescription as description
                """
                result = session.run(query, {'name': name})
                record = result.single()
                
                if record and record['embedding']:
                    cocktail_embedding = record['embedding']
                    similarity = self.calculate_cosine_similarity(question_embedding, cocktail_embedding)
                    cocktail_similarities.append((name, similarity))
                else:
                    # imageDescription_embedding이 없으면 낮은 점수 할당
                    cocktail_similarities.append((name, 0.0))
        
        # 유사도 기준으로 정렬
        cocktail_similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 순위와 점수 출력
        print(f"   → imageDescription 유사도 랭킹:")
        for i, (name, similarity) in enumerate(cocktail_similarities, 1):
            print(f"      {i}. {name} (유사도: {similarity:.3f})")
        
        return [name for name, _ in cocktail_similarities]
    
    def get_cocktail_details(self, cocktail_names: List[str]) -> List[Dict[str, Any]]:
        """칵테일 상세 정보 가져오기 (레시피 중심)"""
        cocktails = []
        with self.driver.session(notifications_disabled_categories=["UNRECOGNIZED", "PERFORMANCE", "DEPRECATION", "HINT"]) as session:
            for name in cocktail_names:
                query = """
                MATCH (c:Cocktail {name: $name})
                OPTIONAL MATCH (c)-[:CATEGORY]->(cat:Category)
                OPTIONAL MATCH (c)-[:HAS_GLASSTYPE]->(g:GlassType)
                OPTIONAL MATCH (c)-[r:HAS_INGREDIENT]->(i:Ingredient)
                RETURN c.name as name, 
                       c.description as description,
                       c.instructions as instructions,
                       c.imageDescription as imageDescription,
                       c.alcoholic as alcoholic,
                       c.ingredients as ingredients_raw,
                       c.ingredientMeasures as measures_raw,
                       cat.name as category,
                       g.name as glassType,
                       collect(DISTINCT {ingredient: i.name, measure: r.measure}) as ingredient_details
                """
                
                result = session.run(query, {'name': name})
                record = result.single()
                
                if record:
                    # ingredients와 measures 파싱
                    ingredients_list = []
                    measures_list = []
                    
                    try:
                        if record['ingredients_raw']:
                            ingredients_list = eval(record['ingredients_raw']) if isinstance(record['ingredients_raw'], str) else record['ingredients_raw']
                        if record['measures_raw']:
                            measures_list = eval(record['measures_raw']) if isinstance(record['measures_raw'], str) else record['measures_raw']
                    except:
                        ingredients_list = []
                        measures_list = []
                    
                    # 재료와 측정값 결합
                    recipe_ingredients = []
                    for i, ingredient in enumerate(ingredients_list):
                        measure = measures_list[i] if i < len(measures_list) else 'unknown'
                        recipe_ingredients.append({
                            'ingredient': ingredient,
                            'measure': measure
                        })
                    
                    cocktails.append({
                        'name': record['name'],
                        'description': record['description'],
                        'instructions': record['instructions'],
                        'imageDescription': record['imageDescription'],
                        'alcoholic': record['alcoholic'],
                        'category': record['category'],
                        'glassType': record['glassType'],
                        'recipe_ingredients': recipe_ingredients,
                        'ingredients': [item['ingredient'] for item in recipe_ingredients]  # 호환성을 위해
                    })
        
        return cocktails
    
    def retrieve(self, user_question: str) -> Dict[str, Any]:
        """Multi-hop 재료 확장 기반 칵테일 검색 알고리즘"""
        print(f"C3 Retrieval (Multi-hop 재료 확장): 사용자 질문 - {user_question}")
        
        # 1단계: 재료와 칵테일 이름 추출
        keywords = self.extract_ingredients_and_cocktails(user_question)
        print(f"1단계 - 키워드 추출: {keywords}")
        
        ingredients = keywords.get('ingredients', [])
        cocktail_names = keywords.get('cocktail_names', [])
        
        all_candidate_cocktails = []
        
        # 2단계: 칵테일 이름이 있으면 직접 검색
        if cocktail_names:
            name_based_cocktails = self.find_cocktails_by_name_similarity(cocktail_names)
            all_candidate_cocktails.extend(name_based_cocktails)
            print(f"2단계 - 칵테일 이름 기반 검색: {len(name_based_cocktails)}개")
        
        # 3단계: 재료가 있으면 Multi-hop 확장 검색
        if ingredients:
            # 3-1: 초기 재료로 칵테일 찾기
            initial_cocktails = self.find_initial_cocktails_by_ingredients(ingredients)
            print(f"3-1단계 - 초기 재료 검색: {len(initial_cocktails)}개")
            
            # 3-2: Multi-hop 확장 검색
            expanded_cocktails = self.multi_hop_ingredient_expansion(ingredients)
            print(f"3-2단계 - Multi-hop 확장: {len(expanded_cocktails)}개")
            
            all_candidate_cocktails.extend(initial_cocktails)
            all_candidate_cocktails.extend(expanded_cocktails)
        
        # 검색 결과가 없는 경우 전체 질문으로 폴백
        if not all_candidate_cocktails:
            print("키워드 검색 결과 없음. 전체 질문 임베딩으로 폴백...")
            question_embedding = self.get_embedding(user_question)
            with self.driver.session(notifications_disabled_categories=["UNRECOGNIZED", "PERFORMANCE", "DEPRECATION", "HINT"]) as session:
                similar_cocktails = self.find_similar_cocktails_by_embedding(
                    question_embedding, session, top_k=self.c3_config['final_top_k']
                )
                all_candidate_cocktails = [name for name, _ in similar_cocktails]
        
        # 4단계: 중복 제거 및 imageDescription 유사도 기반 최종 랭킹
        unique_cocktails = self.remove_duplicates_preserve_order(all_candidate_cocktails)
        print(f"4단계 - 후보 정리: {len(unique_cocktails)}개 (중복 제거 후)")
        
        # imageDescription 임베딩 유사도로 전체 순위 결정
        ranked_cocktails = self.rank_by_image_similarity(user_question, unique_cocktails)
        final_top_k = self.config.get('final_top_k', self.c3_config['final_top_k'])
        print(f"🔧 최종 선정 개수: {final_top_k}개 (config: {self.config.get('final_top_k')}, c3_config: {self.c3_config['final_top_k']})")
        
        # 현재 라운드에 필요한 만큼만 선택
        final_cocktail_names = ranked_cocktails[:final_top_k]
        print(f"4단계 - 최종 선정: {len(final_cocktail_names)}개 (전체 랭킹: {len(ranked_cocktails)}개)")
        for i, name in enumerate(final_cocktail_names, 1):
            print(f"   {i}. {name}")
        
        # 5단계: 상세 정보 가져오기
        final_results = self.get_cocktail_details(final_cocktail_names)
        print(f"5단계 - Multi-hop 검색 완료: {len(final_results)}개 결과")
        
        # dict 형태로 반환 (캐싱을 위해)
        return {
            'results': final_results,
            'full_ranked_names': ranked_cocktails,  # 전체 유사도 랭킹된 이름 리스트
            'current_top_k': final_top_k
        }
