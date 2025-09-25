import pandas as pd
from neo4j import GraphDatabase
import openai
import ast
import os
import json
from typing import List, Dict
import time
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()

class CocktailGraphBuilder:
    def __init__(self, use_python_config: bool = True, config_path: str = "config.json"):
        """
        Initialize the CocktailGraphBuilder with configuration
        """
        if use_python_config:
            # Python 설정 사용
            try:
                from config import get_config
                self.config = get_config()
                print("✅ Python 설정 모듈 사용")
            except ImportError:
                print("⚠️ Python 설정 모듈을 찾을 수 없음. JSON 설정으로 fallback")
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
        else:
            # JSON 설정 사용
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        
        # Get all sensitive data from environment variables
        neo4j_uri = os.getenv('NEO4J_URI')
        neo4j_user = os.getenv('NEO4J_USER')
        neo4j_password = os.getenv('NEO4J_PASSWORD')
        openai_api_key = os.getenv('OPENAI_API_KEY')
        
        if not neo4j_uri:
            raise ValueError("NEO4J_URI not found in environment variables")
        if not neo4j_user:
            raise ValueError("NEO4J_USER not found in environment variables")
        if not neo4j_password:
            raise ValueError("NEO4J_PASSWORD not found in environment variables")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.driver = GraphDatabase.driver(
            neo4j_uri, 
            auth=(neo4j_user, neo4j_password)
        )
        openai.api_key = openai_api_key
        self.embedding_model = self.config['embedding_model']
        self.embedding_cache_file = self.config['embedding_cache_file']
        self.embedding_cache = self._load_embedding_cache()
        self.embedding_dimension = self._determine_embedding_dimension()
        
    def close(self):
        """Close the Neo4j driver connection and save cache"""
        self._save_embedding_cache()
        self.driver.close()
        
    def _load_embedding_cache(self) -> Dict:
        """Load embedding cache from file"""
        if os.path.exists(self.embedding_cache_file):
            try:
                with open(self.embedding_cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading cache: {e}")
                return {}
        return {}
    
    def _save_embedding_cache(self):
        """Save embedding cache to file"""
        try:
            with open(self.embedding_cache_file, 'w') as f:
                json.dump(self.embedding_cache, f, indent=2)
            print(f"Saved embedding cache to {self.embedding_cache_file}")
        except Exception as e:
            print(f"Error saving cache: {e}")

    def _determine_embedding_dimension(self) -> int:
        """Determine embedding dimension for the configured model"""
        try:
            response = openai.embeddings.create(
                input="dimension probe",
                model=self.embedding_model
            )
            return len(response.data[0].embedding)
        except Exception as e:
            raise RuntimeError(
                "Unable to determine embedding dimension from OpenAI embeddings API."
            ) from e
        
    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for text using OpenAI model with caching
        """
        if not text or pd.isna(text):
            return [0.0] * self.embedding_dimension  # Return zero vector for empty text
        
        # Check cache first
        cache_key = f"{self.embedding_model}:{text}"
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
            
        try:
            response = openai.embeddings.create(
                input=text,
                model=self.embedding_model
            )
            embedding = response.data[0].embedding
            # Store in cache
            self.embedding_cache[cache_key] = embedding
            return embedding
        except Exception as e:
            print(f"Error getting embedding for text: {e}")
            return [0.0] * self.embedding_dimension
            
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for multiple texts in batch with caching
        """
        embeddings = []
        texts_to_embed = []
        indices_to_embed = []
        
        # Check cache for each text
        for i, text in enumerate(texts):
            if not text or pd.isna(text):
                embeddings.append([0.0] * self.embedding_dimension)
            else:
                cache_key = f"{self.embedding_model}:{text}"
                if cache_key in self.embedding_cache:
                    embeddings.append(self.embedding_cache[cache_key])
                else:
                    embeddings.append(None)  # Placeholder
                    texts_to_embed.append(text)
                    indices_to_embed.append(i)
        
        # Get embeddings for texts not in cache
        if texts_to_embed:
            print(f"Generating embeddings for {len(texts_to_embed)} texts (using cache for {len(texts) - len(texts_to_embed)} texts)")
            try:
                batch_size = 100
                # Use tqdm for progress bar
                with tqdm(total=len(texts_to_embed), desc="Generating embeddings") as pbar:
                    for i in range(0, len(texts_to_embed), batch_size):
                        batch_texts = texts_to_embed[i:i+batch_size]
                        batch_indices = indices_to_embed[i:i+batch_size]
                        
                        response = openai.embeddings.create(
                            input=batch_texts,
                            model=self.embedding_model
                        )
                        
                        for j, (text, idx) in enumerate(zip(batch_texts, batch_indices)):
                            embedding = response.data[j].embedding
                            embeddings[idx] = embedding
                            # Store in cache
                            cache_key = f"{self.embedding_model}:{text}"
                            self.embedding_cache[cache_key] = embedding
                        
                        pbar.update(len(batch_texts))
                        
                        # Rate limiting
                        time.sleep(0.1)
                    
            except Exception as e:
                print(f"Error getting batch embeddings: {e}")
                # Fill failed embeddings with zero vectors
                for idx in indices_to_embed:
                    if embeddings[idx] is None:
                        embeddings[idx] = [0.0] * self.embedding_dimension
        else:
            print(f"All {len(texts)} embeddings loaded from cache!")
            
        return embeddings
        
    def create_constraints(self):
        """Create unique constraints for the graph"""
        constraints = [
            "CREATE CONSTRAINT cocktail_id IF NOT EXISTS FOR (c:Cocktail) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT ingredient_name IF NOT EXISTS FOR (i:Ingredient) REQUIRE i.name IS UNIQUE",
            "CREATE CONSTRAINT category_name IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE",
            "CREATE CONSTRAINT glass_name IF NOT EXISTS FOR (g:GlassType) REQUIRE g.name IS UNIQUE"
        ]
        
        with self.driver.session() as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                    print(f"Created constraint: {constraint}")
                except Exception as e:
                    print(f"Constraint might already exist: {e}")
                    
    def create_vector_indices(self, create_indices: bool = False):
        """Create vector indices for embeddings (optional)"""
        if not create_indices:
            print("Skipping vector index creation (create_indices=False)")
            return
            
        dimension = self.embedding_dimension
        print(f"Creating vector indices with dimension: {dimension}")
        
        indices = [
            f"""
            CREATE VECTOR INDEX cocktail_name_embedding IF NOT EXISTS
            FOR (c:Cocktail) ON (c.name_embedding)
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: {dimension},
                    `vector.similarity_function`: 'cosine'
                }}
            }}
            """,
            f"""
            CREATE VECTOR INDEX cocktail_imageDescription_embedding IF NOT EXISTS
            FOR (c:Cocktail) ON (c.imageDescription_embedding)
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: {dimension},
                    `vector.similarity_function`: 'cosine'
                }}
            }}
            """,
            f"""
            CREATE VECTOR INDEX ingredient_name_embedding IF NOT EXISTS
            FOR (i:Ingredient) ON (i.name_embedding)
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: {dimension},
                    `vector.similarity_function`: 'cosine'
                }}
            }}
            """,
            f"""
            CREATE VECTOR INDEX category_name_embedding IF NOT EXISTS
            FOR (c:Category) ON (c.name_embedding)
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: {dimension},
                    `vector.similarity_function`: 'cosine'
                }}
            }}
            """,
            f"""
            CREATE VECTOR INDEX glasstype_name_embedding IF NOT EXISTS
            FOR (g:GlassType) ON (g.name_embedding)
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: {dimension},
                    `vector.similarity_function`: 'cosine'
                }}
            }}
            """
        ]
        
        with self.driver.session() as session:
            for i, index in enumerate(indices):
                try:
                    session.run(index)
                    index_names = [
                        "cocktail_name_embedding",
                        "cocktail_imageDescription_embedding", 
                        "ingredient_name_embedding",
                        "category_name_embedding",
                        "glasstype_name_embedding"
                    ]
                    print(f"Created vector index: {index_names[i]}")
                except Exception as e:
                    print(f"Vector index might already exist: {e}")
                    
    def preprocess_data(self, csv_path: str) -> pd.DataFrame:
        """Load and preprocess the cocktail data"""
        # Try utf-8 first, then latin-1 if it fails
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding='latin-1')
        
        # 전체 데이터 사용
        print(f"Processing all {len(df)} rows...")
        
        # Fill NaN values with empty strings
        df = df.fillna('')
        
        # Parse ingredients and measures
        def safe_parse(value):
            if not value:
                return []
            try:
                return ast.literal_eval(value)
            except:
                return []
                
        df['ingredients_parsed'] = df['ingredients'].apply(safe_parse)
        df['measures_parsed'] = df['ingredientMeasures'].apply(safe_parse)
        
        return df
        
    def import_cocktails(self, df: pd.DataFrame):
        """Import cocktail nodes with embeddings"""
        print("Generating embeddings for cocktail names...")
        names = df['name'].tolist()
        name_embeddings = self.get_embeddings_batch(names)
        
        print("Generating embeddings for imageDescription...")
        image_descriptions = df['imageDescription'].tolist() if 'imageDescription' in df.columns else [''] * len(df)
        image_description_embeddings = self.get_embeddings_batch(image_descriptions)
        
        with self.driver.session() as session:
            # Use tqdm for progress bar
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Importing cocktails"):
                # Prepare cocktail data
                cocktail_data = {
                    'id': int(row['id']),
                    'name': row['name'],
                    'name_embedding': name_embeddings[idx],
                    'alcoholic': row['alcoholic'],
                    'ingredients': row['ingredients'],
                    'drinkThumbnail': row['drinkThumbnail'],
                    'ingredientMeasures': row['ingredientMeasures'],
                    'description': row['desciription'],
                    'instructions': row['instructions'],
                    'imageDescription': row.get('imageDescription', ''),
                    'imageDescription_embedding': image_description_embeddings[idx]
                }
                
                # Create cocktail node
                query = """
                MERGE (c:Cocktail {id: $id})
                SET c.name = $name,
                    c.name_embedding = $name_embedding,
                    c.alcoholic = $alcoholic,
                    c.ingredients = $ingredients,
                    c.drinkThumbnail = $drinkThumbnail,
                    c.ingredientMeasures = $ingredientMeasures,
                    c.description = $description,
                    c.instructions = $instructions,
                    c.imageDescription = $imageDescription,
                    c.imageDescription_embedding = $imageDescription_embedding
                """
                
                session.run(query, cocktail_data)
                    
        print(f"\nImported {len(df)} cocktails successfully!")
        
    def import_ingredients_and_relationships(self, df: pd.DataFrame):
        """Import ingredients, categories, glass types and create relationships"""
        # Pre-collect all unique values to generate embeddings in batch
        all_categories = set()
        all_glass_types = set()
        all_ingredients = set()
        
        print("Collecting unique values...")
        for _, row in df.iterrows():
            if row['category']:
                all_categories.add(row['category'])
            if row['glassType']:
                all_glass_types.add(row['glassType'])
            for ingredient in row['ingredients_parsed']:
                if ingredient:
                    all_ingredients.add(ingredient.lower().strip())
        
        # Generate embeddings for all unique values (name 속성만 임베딩)
        print(f"Generating embeddings for {len(all_categories)} categories...")
        category_list = list(all_categories)
        category_embeddings = self.get_embeddings_batch(category_list)
        category_embedding_dict = dict(zip(category_list, category_embeddings))
        
        print(f"Generating embeddings for {len(all_glass_types)} glass types...")
        glass_type_list = list(all_glass_types)
        glass_type_embeddings = self.get_embeddings_batch(glass_type_list)
        glass_type_embedding_dict = dict(zip(glass_type_list, glass_type_embeddings))
        
        print(f"Generating embeddings for {len(all_ingredients)} ingredients...")
        ingredient_list = list(all_ingredients)
        ingredient_embeddings = self.get_embeddings_batch(ingredient_list)
        ingredient_embedding_dict = dict(zip(ingredient_list, ingredient_embeddings))
        
        with self.driver.session() as session:
            # First, create all nodes with embeddings
            print("Creating category nodes with embeddings...")
            for category in tqdm(all_categories, desc="Creating categories"):
                session.run("""
                    MERGE (cat:Category {name: $category_name})
                    SET cat.name_embedding = $name_embedding
                """, {
                    'category_name': category,
                    'name_embedding': category_embedding_dict[category]
                })
            
            print("Creating glass type nodes with embeddings...")
            for glass_type in tqdm(all_glass_types, desc="Creating glass types"):
                session.run("""
                    MERGE (g:GlassType {name: $glass_name})
                    SET g.name_embedding = $name_embedding
                """, {
                    'glass_name': glass_type,
                    'name_embedding': glass_type_embedding_dict[glass_type]
                })
            
            print("Creating ingredient nodes with embeddings...")
            for ingredient in tqdm(all_ingredients, desc="Creating ingredients"):
                session.run("""
                    MERGE (ing:Ingredient {name: $ingredient_name})
                    SET ing.name_embedding = $name_embedding
                """, {
                    'ingredient_name': ingredient,
                    'name_embedding': ingredient_embedding_dict[ingredient]
                })
            
            # Now create relationships using MERGE
            print("Creating relationships...")
            for _, row in tqdm(df.iterrows(), total=len(df), desc="Creating relationships"):
                cocktail_id = int(row['id'])
                
                # Create Category relationship using MERGE
                if row['category']:
                    session.run("""
                        MATCH (c:Cocktail {id: $cocktail_id})
                        MATCH (cat:Category {name: $category_name})
                        MERGE (c)-[:CATEGORY]->(cat)
                    """, {'cocktail_id': cocktail_id, 'category_name': row['category']})
                
                # Create GlassType relationship using MERGE
                if row['glassType']:
                    session.run("""
                        MATCH (c:Cocktail {id: $cocktail_id})
                        MATCH (g:GlassType {name: $glass_name})
                        MERGE (c)-[:HAS_GLASSTYPE]->(g)
                    """, {'cocktail_id': cocktail_id, 'glass_name': row['glassType']})
                
                # Create Ingredient relationships using MERGE
                ingredients = row['ingredients_parsed']
                measures = row['measures_parsed']
                
                for i, ingredient in enumerate(ingredients):
                    if ingredient:
                        measure = measures[i] if i < len(measures) else 'unknown'
                        
                        session.run("""
                            MATCH (c:Cocktail {id: $cocktail_id})
                            MATCH (ing:Ingredient {name: $ingredient_name})
                            MERGE (c)-[r:HAS_INGREDIENT]->(ing)
                            SET r.measure = $measure
                        """, {
                            'cocktail_id': cocktail_id,
                            'ingredient_name': ingredient.lower().strip(),  # Normalize ingredient names
                            'measure': measure
                        })
                        
        print(f"\nCreated all nodes and relationships successfully!")
        
    def build_graph(self, csv_path: str, create_vector_indices: bool = False):
        """Main method to build the entire graph"""
        print("Starting cocktail graph construction...")
        
        # Create constraints and indices
        print("\n1. Creating constraints...")
        self.create_constraints()
        
        print("\n2. Creating vector indices...")
        self.create_vector_indices(create_indices=create_vector_indices)
        
        # Preprocess data
        print("\n3. Preprocessing data...")
        df = self.preprocess_data(csv_path)
        print(f"Loaded {len(df)} cocktails")
        
        # Import cocktails
        print("\n4. Importing cocktails with embeddings...")
        self.import_cocktails(df)
        
        # Import ingredients and relationships
        print("\n5. Creating relationships...")
        self.import_ingredients_and_relationships(df)
        
        print("\nGraph construction completed!")
        
    def verify_graph(self):
        """Verify the graph structure with sample queries"""
        with self.driver.session() as session:
            # Count nodes - separate queries to avoid UNION column name issues
            print("\nNode counts:")
            
            result = session.run("MATCH (c:Cocktail) RETURN count(c) as count")
            print(f"  Cocktails: {result.single()['count']}")
            
            result = session.run("MATCH (i:Ingredient) RETURN count(i) as count")
            print(f"  Ingredients: {result.single()['count']}")
            
            result = session.run("MATCH (cat:Category) RETURN count(cat) as count")
            print(f"  Categories: {result.single()['count']}")
            
            result = session.run("MATCH (g:GlassType) RETURN count(g) as count")
            print(f"  Glass Types: {result.single()['count']}")
            
                
            # Count relationships - separate queries
            print("\nRelationship counts:")
            
            result = session.run("MATCH ()-[r:HAS_INGREDIENT]->() RETURN count(r) as count")
            print(f"  HAS_INGREDIENT: {result.single()['count']}")
            
            result = session.run("MATCH ()-[r:CATEGORY]->() RETURN count(r) as count")
            print(f"  CATEGORY: {result.single()['count']}")
            
            result = session.run("MATCH ()-[r:HAS_GLASSTYPE]->() RETURN count(r) as count")
            print(f"  HAS_GLASSTYPE: {result.single()['count']}")
                
            # Sample cocktail with all relationships
            results = session.run("""
                MATCH (c:Cocktail)-[:HAS_INGREDIENT]->(i:Ingredient)
                WHERE c.name = '151 florida bushwacker'
                RETURN c.name as cocktail, collect(i.name) as ingredients
                LIMIT 1
            """)
            
            print("\nSample cocktail:")
            for record in results:
                print(f"  Cocktail: {record['cocktail']}")
                print(f"  Ingredients: {record['ingredients']}")


if __name__ == "__main__":
    # Create graph builder with Python config (fallback to JSON if needed)
    builder = CocktailGraphBuilder(use_python_config=True)
    
    try:
        # Build the graph (벡터 인덱스 생성하지 않음 - 원래 방식)
        builder.build_graph("cocktail_data_436_final.csv", create_vector_indices=False)
        
        # Verify the graph
        builder.verify_graph()
        
    finally:
        builder.close()
