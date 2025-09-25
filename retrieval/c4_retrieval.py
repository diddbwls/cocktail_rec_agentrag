from typing import List, Dict, Any
import sys
import os

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.config import get_config, get_c4_config
from retrieval.base_retrieval import BaseRetrieval

class C4Retrieval(BaseRetrieval):
    """C4 태스크: 재료기반 유사 레시피 칵테일 추천"""
    
    def __init__(self, use_python_config: bool = True):
        """Initialize C4 retrieval system"""
        if use_python_config:
            # Python 설정 사용
            config = get_config()
            c4_config = get_c4_config()
        else:
            # 기존 JSON 설정 사용 (하위 호환성)
            import json
            with open("config.json", 'r') as f:
                config = json.load(f)
            c4_config = config['c4_config']
        
        # 기본 클래스 초기화
        super().__init__(config, c4_config)
        self.c4_config = c4_config  # 편의를 위해 별도 저장
    
    def extract_target_cocktail(self, user_question: str) -> Dict[str, Any]:
        """LLM을 사용하여 사용자 질문에서 타겟 칵테일 정보 추출"""
        result = super().extract_keywords(user_question)
        
        # C4 특화: 필수 키 확인
        if 'target_cocktail' not in result:
            result['target_cocktail'] = ""
        if 'ingredients' not in result:
            result['ingredients'] = []
                
        return result
    
    def find_target_cocktail_by_name(self, cocktail_name: str) -> str:
        """칵테일 이름으로 정확한 타겟 찾기"""
        if not cocktail_name:
            return ""
        
        with self.driver.session() as session:
            # 정확한 이름 매치 먼저
            query = """
            MATCH (c:Cocktail)
            WHERE toLower(c.name) CONTAINS toLower($name)
            RETURN c.name as name
            LIMIT 1
            """
            
            result = session.run(query, {'name': cocktail_name})
            record = result.single()
            
            if record:
                print(f"   → 타겟 칵테일 발견: {record['name']}")
                return record['name']
            
            # 임베딩 기반 유사 검색으로 폴백
            name_embedding = self.get_embedding(cocktail_name)
            similar_cocktails = self.find_similar_cocktails_by_embedding(name_embedding, session, top_k=1)
            
            if similar_cocktails and similar_cocktails[0][1] > self.c4_config['name_similarity_threshold']:
                target_name = similar_cocktails[0][0]
                print(f"   → 유사 이름으로 타겟 발견: {target_name}")
                return target_name
        
        print(f"   → 타겟 칵테일 '{cocktail_name}' 찾을 수 없음")
        return ""
    
    def find_target_by_ingredients(self, ingredients: List[str]) -> str:
        """재료 조합으로 가장 적합한 타겟 칵테일 찾기"""
        if not ingredients:
            return ""
        
        with self.driver.session() as session:
            # 재료 매치 점수가 높은 칵테일 찾기
            query = """
            MATCH (c:Cocktail)-[:HAS_INGREDIENT]->(i:Ingredient)
            WHERE i.name IN $ingredients
            WITH c, count(i) as matched_count
            
            // 총 재료 개수 계산
            MATCH (c)-[:HAS_INGREDIENT]->(all_ingredients:Ingredient)
            WITH c, matched_count, count(all_ingredients) as total_ingredients
            WHERE matched_count >= 2  // 최소 2개 재료 매치
            RETURN c.name as name, 
                   matched_count, 
                   total_ingredients,
                   toFloat(matched_count) / total_ingredients as match_ratio
            ORDER BY match_ratio DESC, matched_count DESC
            LIMIT 1
            """
            
            result = session.run(query, {'ingredients': ingredients})
            record = result.single()
            
            if record:
                target_name = record['name']
                print(f"   → 재료 기반 타겟 발견: {target_name} (매치율: {record['match_ratio']:.2f})")
                return target_name
        
        print("   → 재료 기반 타겟 찾을 수 없음")
        return ""
    
    def relationship_based_search(self, target_cocktail: str) -> List[Dict[str, Any]]:
        """동일 재료로 향하는 관계 수로 계산"""
        if not target_cocktail:
            return []
        
        with self.driver.session() as session:
            # 관계 기반 유사도 계산
            query = """
            // 타겟 칵테일의 재료들 찾기
            MATCH (c1:Cocktail {name: $target})-[:HAS_INGREDIENT]->(i:Ingredient)
            WITH c1, collect(i) as target_ingredients, count(i) as target_count
            
            // 다른 칵테일들과 공유하는 재료 관계 계산
            UNWIND target_ingredients as ingredient
            MATCH (c2:Cocktail)-[:HAS_INGREDIENT]->(ingredient)
            WHERE c2 <> c1
            WITH c1, c2, target_count, count(ingredient) as shared_relationships
            
            // c2의 재료 개수를 별도로 계산
            MATCH (c2)-[:HAS_INGREDIENT]->(i2:Ingredient)
            WITH c1, c2, target_count, shared_relationships, count(i2) as c2_ingredient_count
            WHERE abs(c2_ingredient_count - target_count) <= $complexity_tolerance  // 복잡도 필터링
            AND shared_relationships >= $min_shared  // 최소 공유 관계 수
            
            // 공유 재료 이름들 수집
            MATCH (c1)-[:HAS_INGREDIENT]->(shared_ingredient:Ingredient)<-[:HAS_INGREDIENT]-(c2)
            WITH c1, c2, shared_relationships, c2_ingredient_count,
                 collect(DISTINCT shared_ingredient.name) as shared_ingredients
            
            // 카테고리와 글라스 정보 가져오기
            OPTIONAL MATCH (c2)-[:CATEGORY]->(cat2:Category)
            OPTIONAL MATCH (c2)-[:HAS_GLASSTYPE]->(g2:GlassType)
            
            RETURN c2.name as name,
                   shared_relationships,
                   shared_ingredients,
                   cat2.name as category,
                   g2.name as glassType,
                   c2_ingredient_count
            ORDER BY shared_relationships DESC, c2_ingredient_count ASC
            LIMIT $top_k
            """
            
            result = session.run(query, {
                'target': target_cocktail,
                'complexity_tolerance': self.c4_config['complexity_tolerance'],
                'min_shared': self.c4_config['min_shared_ingredients'],
                'top_k': self.c4_config['initial_top_k']
            })
            
            similar_cocktails = []
            for record in result:
                similar_cocktails.append({
                    'name': record['name'],
                    'shared_relationships': record['shared_relationships'],
                    'shared_ingredients': record['shared_ingredients'],
                    'category': record['category'],
                    'glassType': record['glassType'],
                    'ingredient_count': record['c2_ingredient_count']
                })
            
            print(f"   → 관계 기반 검색: {len(similar_cocktails)}개 칵테일")
            return similar_cocktails
    
    def find_similar_cocktails_by_embedding(self, embedding: List[float], session, top_k: int = 5):
        """임베딩 기반 칵테일 유사도 검색 (폴백용)"""
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
                       c.ingredients as ingredients_raw,
                       c.ingredientMeasures as measures_raw,
                       cat.name as category,
                       g.name as glassType,
                       collect(DISTINCT i.name) as ingredients
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
                        'ingredients': record['ingredients']  # 그래프에서 직접
                    })
        
        return cocktails
    
    def retrieve(self, user_question: str) -> List[Dict[str, Any]]:
        """관계 기반 + 복잡도 필터링 칵테일 대안 검색 알고리즘"""
        print(f"C4 Retrieval (관계 기반 + 복잡도): 사용자 질문 - {user_question}")
        
        # 1단계: 타겟 칵테일 정보 추출
        target_info = self.extract_target_cocktail(user_question)
        print(f"1단계 - 타겟 정보 추출: {target_info}")
        
        target_cocktail_name = target_info.get('target_cocktail', '')
        ingredients = target_info.get('ingredients', [])
        # 복잡도 허용 범위는 config에서 설정됨
        
        # 2단계: 타겟 칵테일 결정
        target_cocktail = ""
        if target_cocktail_name:
            target_cocktail = self.find_target_cocktail_by_name(target_cocktail_name)
        
        if not target_cocktail and ingredients:
            target_cocktail = self.find_target_by_ingredients(ingredients)
        
        if not target_cocktail:
            print("❌ 타겟 칵테일을 결정할 수 없습니다.")
            # 폴백: 전체 질문으로 임베딩 검색
            question_embedding = self.get_embedding(user_question)
            with self.driver.session() as session:
                similar_cocktails = self.find_similar_cocktails_by_embedding(
                    question_embedding, session, top_k=self.c4_config['embedding_fallback_top_k']
                )
                fallback_names = [name for name, _ in similar_cocktails]
                return self.get_cocktail_details(fallback_names)
        
        print(f"2단계 - 타겟 칵테일 결정: {target_cocktail}")
        
        # 2.5단계: 타겟 칵테일 정보 가져오기
        target_details = self.get_cocktail_details([target_cocktail])
        if target_details:
            target_info = target_details[0]
            print(f"\n🎯 타겟 칵테일 정보:")
            print(f"   • 이름: {target_info['name']}")
            print(f"   • 카테고리: {target_info.get('category', 'N/A')}")
            print(f"   • 글라스: {target_info.get('glassType', 'N/A')}")
            recipe_ingredients = target_info.get('recipe_ingredients', [])
            if recipe_ingredients:
                print(f"   • 재료 ({len(recipe_ingredients)}개):")
                for ingredient_info in recipe_ingredients:
                    measure = ingredient_info.get('measure', 'unknown')
                    ingredient = ingredient_info.get('ingredient', 'unknown')
                    print(f"     - {measure} {ingredient}")
            print()
        
        # 3단계: 관계 기반 유사 칵테일 검색
        similar_cocktails = self.relationship_based_search(target_cocktail)
        print(f"3단계 - 관계 기반 검색: {len(similar_cocktails)}개 결과")
        
        # 4단계: 상위 결과 선택 (동적으로 업데이트된 값 사용)
        final_top_k = self.config.get('final_top_k', self.c4_config['final_top_k'])
        print(f"🔧 최종 선정 개수: {final_top_k}개 (config: {self.config.get('final_top_k')}, c4_config: {self.c4_config['final_top_k']})")
        final_cocktail_names = [item['name'] for item in similar_cocktails[:final_top_k]]
        print(f"4단계 - 최종 선정: {len(final_cocktail_names)}개")
        
        for i, item in enumerate(similar_cocktails[:final_top_k], 1):
            shared_count = len(item['shared_ingredients'])
            print(f"   {i}. {item['name']} (공유재료: {shared_count}개)")
        
        # 5단계: 상세 정보 가져오기
        final_results = self.get_cocktail_details(final_cocktail_names)
        print(f"5단계 - 관계 기반 검색 완료: {len(final_results)}개 결과")
        
        # 타겟 칵테일 정보를 0번째로 추가
        if target_details:
            target_details[0]['is_target'] = True  # 타겟임을 표시
            final_results = target_details + final_results
        
        return final_results



# test_queries = [
#     "Manhattan과 유사한 칵테일 추천해줘",
#     "Mojito 같은 스타일의 다른 칵테일들",
#     "위스키와 베르무스를 사용하는 칵테일과 비슷한 레시피",
#     "복잡하지 않은 간단한 칵테일로 Martini 대안"
# ]
