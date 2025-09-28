"""
Simple similarity-based retrieval for ablation study
Compares query embeddings with cocktail imageDescription embeddings
"""
import json
import openai
import numpy as np
from typing import List, Dict, Any
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

class SimpleRetrieval:
    """Simple embedding similarity-based retrieval using Neo4j"""
    
    def __init__(self, embedding_model: str = "text-embedding-3-small"):
        self.embedding_model = embedding_model
        openai.api_key = os.getenv('OPENAI_API_KEY')
        
        # Initialize Neo4j connection
        neo4j_uri = os.getenv('NEO4J_URI')
        neo4j_user = os.getenv('NEO4J_USER')
        neo4j_password = os.getenv('NEO4J_PASSWORD')
        
        if not all([neo4j_uri, neo4j_user, neo4j_password]):
            raise ValueError("Missing required Neo4j environment variables")
            
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
    
    def _get_all_cocktails_with_embeddings(self) -> List[Dict[str, Any]]:
        """Get all cocktails with their imageDescription embeddings from Neo4j"""
        with self.driver.session() as session:
            query = """
            MATCH (c:Cocktail)
            OPTIONAL MATCH (c)-[:CATEGORY]->(cat:Category)
            OPTIONAL MATCH (c)-[:HAS_GLASSTYPE]->(g:GlassType)
            RETURN c.name as name,
                   c.description as description,
                   c.imageDescription as imageDescription,
                   c.imageDescription_embedding as imageDescription_embedding,
                   c.instructions as instructions,
                   c.alcoholic as alcoholic,
                   c.ingredients as ingredients,
                   cat.name as category,
                   g.name as glassType
            """
            result = session.run(query)
            cocktails = []
            for record in result:
                # Handle ingredients - convert to proper list if it's a string
                ingredients = record['ingredients']
                if isinstance(ingredients, str):
                    # If it's a JSON string, parse it
                    try:
                        ingredients = json.loads(ingredients)
                    except:
                        # If parsing fails, split by comma and clean
                        ingredients = [ing.strip() for ing in ingredients.split(',') if ing.strip()]
                elif not ingredients:
                    ingredients = []
                
                cocktails.append({
                    'name': record['name'],
                    'description': record['description'],
                    'imageDescription': record['imageDescription'],
                    'imageDescription_embedding': record['imageDescription_embedding'],
                    'instructions': record['instructions'],
                    'alcoholic': record['alcoholic'],
                    'ingredients': ingredients,
                    'category': record['category'],
                    'glassType': record['glassType']
                })
            return cocktails
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text"""
        try:
            response = openai.embeddings.create(
                input=text,
                model=self.embedding_model
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return [0.0] * 1536  # Default dimension for text-embedding-3-small
    
    def calculate_cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        if not embedding1 or not embedding2:
            return 0.0
        
        return np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
    
    def retrieve(self, user_question: str, top_k: int = 6) -> List[Dict[str, Any]]:
        """
        Retrieve cocktails based on simple embedding similarity
        
        Args:
            user_question: User query
            top_k: Number of results to return
            
        Returns:
            List of cocktail dictionaries with similarity scores
        """
        # Get query embedding
        query_embedding = self.get_embedding(user_question)
        
        # Get all cocktails with embeddings from Neo4j
        cocktails = self._get_all_cocktails_with_embeddings()
        
        # Calculate similarities
        similarities = []
        
        for cocktail in cocktails:
            # Get imageDescription embedding
            cocktail_embedding = cocktail.get('imageDescription_embedding')
            
            if cocktail_embedding and query_embedding:
                similarity = self.calculate_cosine_similarity(query_embedding, cocktail_embedding)
                
                similarities.append({
                    'name': cocktail['name'],
                    'description': cocktail.get('description', ''),
                    'imageDescription': cocktail.get('imageDescription', ''),
                    'category': cocktail.get('category', ''),
                    'glassType': cocktail.get('glassType', ''),
                    'ingredients': cocktail.get('ingredients', []),
                    'instructions': cocktail.get('instructions', ''),
                    'alcoholic': cocktail.get('alcoholic', ''),
                    'similarity_score': similarity
                })
        
        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
        return similarities[:top_k]
    
    def format_cocktails_as_context(self, cocktails: List[Dict[str, Any]]) -> str:
        """Format cocktails as context string"""
        context_parts = []
        for i, cocktail in enumerate(cocktails, 1):
            name = cocktail.get('name', 'Unknown')
            category = cocktail.get('category', 'Unknown')
            glass_type = cocktail.get('glassType', 'Unknown')
            ingredients = cocktail.get('ingredients', [])
            
            context_part = f"{i}. **{name}**\n"
            context_part += f"   - category: {category}\n"
            context_part += f"   - glass_type: {glass_type}\n"
            context_part += f"   - ingredients: {', '.join(ingredients) if ingredients else 'Unknown'}\n"
            context_parts.append(context_part)
        
        return "\n".join(context_parts)