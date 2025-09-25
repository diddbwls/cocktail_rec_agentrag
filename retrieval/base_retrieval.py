"""
칵테일 검색 시스템 기본 클래스
모든 C1~C4 retrieval 시스템의 공통 기능을 제공
"""
import json
import openai
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
from typing import List, Dict, Any
import numpy as np
from abc import ABC, abstractmethod

# Load environment variables
load_dotenv()

class BaseRetrieval(ABC):
    """모든 칵테일 검색 시스템의 기본 클래스"""
    
    def __init__(self, config: Dict[str, Any], task_config: Dict[str, Any]):
        """기본 초기화"""
        self.config = config
        self.task_config = task_config
        
        # Initialize Neo4j connection
        neo4j_uri = os.getenv('NEO4J_URI')
        neo4j_user = os.getenv('NEO4J_USER')
        neo4j_password = os.getenv('NEO4J_PASSWORD')
        openai_api_key = os.getenv('OPENAI_API_KEY')
        
        if not all([neo4j_uri, neo4j_user, neo4j_password, openai_api_key]):
            raise ValueError("Missing required environment variables")

        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        openai.api_key = openai_api_key
        self.embedding_dimension = self._determine_embedding_dimension()
        
        # Cache for categories
        self.categories_cache = None
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
    
    def get_all_categories(self) -> List[str]:
        """DB에서 모든 카테고리 이름 가져오기"""
        if self.categories_cache is not None:
            return self.categories_cache
            
        with self.driver.session() as session:
            result = session.run("MATCH (c:Category) RETURN c.name as name")
            categories = [record['name'] for record in result]
            self.categories_cache = categories
            return categories
    
    def get_embedding(self, text: str) -> List[float]:
        """텍스트의 임베딩 생성"""
        try:
            response = openai.embeddings.create(
                input=text,
                model=self.config['embedding_model']
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return self._zero_vector()

    def _determine_embedding_dimension(self) -> int:
        """Determine embedding dimension for the configured model"""
        try:
            response = openai.embeddings.create(
                input="dimension probe",
                model=self.config['embedding_model']
            )
            return len(response.data[0].embedding)
        except Exception as e:
            raise RuntimeError(
                "Unable to determine embedding dimension from OpenAI embeddings API."
            ) from e

    def _zero_vector(self) -> List[float]:
        return [0.0] * self.embedding_dimension
    
    def extract_keywords(self, user_question: str) -> Dict[str, List[str]]:
        """LLM을 사용하여 사용자 질문에서 키워드 추출"""
        categories = self.get_all_categories()
        category_list = ", ".join(categories)
        
        try:
            prompt = self.task_config['keyword_extraction_prompt'].format(
                user_question=user_question,
                category_list=category_list
            )
        except KeyError as e:
            print(f"❌ Prompt formatting error: {e}")
            print(f"Available template: {self.task_config['keyword_extraction_prompt']}")
            raise e
        
        try:
            system_message = self.task_config.get('system_message', "You are a keyword extraction expert. Always respond with valid JSON only.")
            
            response = openai.chat.completions.create(
                model=self.config['model'],
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config['temperature']
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            print(f"Error extracting keywords: {e}")
            return self._get_default_keywords()
    
    def _get_default_keywords(self) -> Dict[str, List[str]]:
        """기본 키워드 구조 반환 (각 서브클래스에서 필요시 오버라이드)"""
        return {
            "cocktail": [], 
            "include_ingredients": [], 
            "exclude_ingredients": [], 
            "glassType": [], 
            "category": [], 
            "visual_keywords": []
        }
    
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
                        'ingredients': record['ingredients']  # 그래프에서 직접 가져온 재료들
                    })
        
        return cocktails
    
    def calculate_cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """코사인 유사도 계산"""
        if not embedding1 or not embedding2:
            return 0.0
        
        return np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
    
    @abstractmethod
    def retrieve(self, user_question: str) -> List[Dict[str, Any]]:
        """각 검색 시스템별 구체적인 검색 로직 (서브클래스에서 구현)"""
        pass