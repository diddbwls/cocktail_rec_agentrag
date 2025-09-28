import pandas as pd
import openai
import json
import os
from typing import Dict
import time
from tqdm import tqdm


# Configuration
# Target CSV filename to evaluate
CSV_FILENAME = "gpt-4o-mini_ablation_simple_rag.csv" 

# Running this script will save evaluation results and summary files to the result folder
# Output format:
# Persuasiveness: score (±std_deviation)
# Transparency: score (±std_deviation) 
# Accuracy: score (±std_deviation)
# Satisfaction: score (±std_deviation)
# Overall Average: score (±std_deviation)



class LLMAsJudge:
    def __init__(self, api_key: str = None):
        self.client = openai.OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
        
    def evaluate_answer(self, final_answer: str, query: str, context: str = "") -> Dict[str, int]:
        prompt = f"""
You are a customer using a cocktail recommendation platform.
The system has suggested a cocktail to you, along with an explanation text.
Your task is to evaluate the quality of this explanation from the perspective of an end user. 
Judge the explanation strictly based on the provided query and context. 
Do not reward extra details if they are not supported by the context.

Please evaluate on the following four criteria, giving each a score between 1 and 100:

1. **Persuasiveness (1-100)**: "This explanation is convincing to me."
   - Does it sound compelling and trustworthy from a user perspective?
   - Logical reasoning should be rewarded, not just length or flowery language.

2. **Transparency (1-100)**: "Based on this explanation, I understand why this cocktail is recommended."
   - As a user, can you clearly see the connection between the query, the context, and the final answer?
   - The reasoning should be explicit and easy to follow.

3. **Accuracy (1-100)**: "This explanation is consistent with the provided query and context."
   - Check if the explanation correctly uses the given context.
   - If the explanation mentions details (ingredients, colors, garnishes, preparation methods) not present in the context, lower the score significantly.

4. **Satisfaction (1-100)**: "I am satisfied with this explanation."
   - Overall impression as a user: does this explanation meet your needs and answer your query?
   - Balance clarity, correctness, and persuasiveness.

Return your evaluation strictly in the following JSON format, with no additional text:
{{
    "persuasiveness": <score>,
    "transparency": <score>, 
    "accuracy": <score>,
    "satisfaction": <score>
}}

Query: {query}
Context: {context}
Final Answer (explanation): {final_answer}

"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200
            )
            
            result = json.loads(response.choices[0].message.content.strip())
            return result
            
        except Exception as e:
            print(f"Error evaluating answer: {e}")
            return {"persuasiveness": 0, "transparency": 0, "accuracy": 0, "satisfaction": 0}
    
    def evaluate_csv(self, csv_path: str, output_path: str = None) -> pd.DataFrame:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        
        if output_path is None:
            filename = CSV_FILENAME.replace('.csv', '')
            output_path = f"/Users/yujin/Desktop/cocktail_rec_agentrag/ablation_llm+rag/result/{filename}.csv"
        
        evaluation_results = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating answers"):
            final_answer = str(row['final_answer']) if pd.notna(row['final_answer']) else ""
            query = str(row.get('query_EN', '')) if pd.notna(row.get('query_EN')) else ""
            context = str(row.get('final_context', '')) if pd.notna(row.get('final_context')) else ""
            
            if not final_answer.strip():
                scores = {"persuasiveness": 0, "transparency": 0, "accuracy": 0, "satisfaction": 0}
            else:
                scores = self.evaluate_answer(final_answer, query, context)
            
            evaluation_results.append(scores)
            time.sleep(0.1)
        
        for criterion in ["persuasiveness", "transparency", "accuracy", "satisfaction"]:
            df[f"{criterion}_score"] = [result[criterion] for result in evaluation_results]
        
        df['average_score'] = df[['persuasiveness_score', 'transparency_score', 'accuracy_score', 'satisfaction_score']].mean(axis=1)
        
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"Evaluation completed. Results saved to: {output_path}")
        
        return df
    
    def generate_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        score_columns = ['persuasiveness_score', 'transparency_score', 'accuracy_score', 'satisfaction_score']
        
        summary_data = {
            'Metric': score_columns + ['average_score'],
            'Mean_Score': [df[col].mean() for col in score_columns] + [df['average_score'].mean()],
            'Std_Score': [df[col].std() for col in score_columns] + [df['average_score'].std()],
            'Min_Score': [df[col].min() for col in score_columns] + [df['average_score'].min()],
            'Max_Score': [df[col].max() for col in score_columns] + [df['average_score'].max()],
            'Total_Samples': [len(df)] * (len(score_columns) + 1)
        }
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save summary to result directory
        filename = CSV_FILENAME.replace('.csv', '')
        summary_path = f"/Users/yujin/Desktop/cocktail_rec_agentrag/ablation_llm+rag/result/{filename}_summary.csv"
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        print(f"Summary saved to: {summary_path}")
        
        return summary_df

def main():
    csv_path = f"/Users/yujin/Desktop/cocktail_rec_agentrag/ablation_llm+rag/output/{CSV_FILENAME}"
    judge = LLMAsJudge()
    
    # Start timing
    start_time = time.time()
    print(f"Starting evaluation of: {CSV_FILENAME}")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    
    try:
        evaluated_df = judge.evaluate_csv(csv_path)
        
        # Generate summary
        summary_df = judge.generate_summary(evaluated_df)
        
        # Calculate elapsed time
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print(f"\n=== Evaluation Completed ===")
        print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
        print(f"Total elapsed time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        print(f"Total samples evaluated: {len(evaluated_df)}")
        
        print("\n=== Summary Statistics ===")
        score_columns = ['persuasiveness_score', 'transparency_score', 'accuracy_score', 'satisfaction_score']
        for col in score_columns:
            mean_score = evaluated_df[col].mean()
            std_score = evaluated_df[col].std()
            print(f"{col.replace('_score', '').title()}: {mean_score:.2f} (±{std_score:.2f})")
        
        overall_avg = evaluated_df['average_score'].mean()
        overall_std = evaluated_df['average_score'].std()
        print(f"Overall Average: {overall_avg:.2f} (±{overall_std:.2f})")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")

if __name__ == "__main__":
    main()