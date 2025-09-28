import pandas as pd
import openai
import json
import os
import sys
from typing import Dict
import time
from tqdm import tqdm

# Add parent directory to path for importing prompts
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from prompts.llm_judge_prompt import get_evaluation_prompt


# Configuration - Hyperparameters
CSV_FILENAME = "gpt-4o-mini_ablation_simple_rag.csv"  # Target CSV filename to evaluate
SAMPLE_LIMIT = None  # Set to integer to limit number of samples (None = all samples)
TEMPERATURE = 0.0  # OpenAI API temperature (0.0-1.0)
MAX_TOKENS = 100  # Maximum tokens for LLM response
API_DELAY = 0.1  # Delay between API calls in seconds 

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
        prompt = get_evaluation_prompt(final_answer, query, context)
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS
            )
            
            result = json.loads(response.choices[0].message.content.strip())
            return result
            
        except Exception as e:
            print(f"Error evaluating answer: {e}")
            return {"persuasiveness": 1, "transparency": 1, "accuracy": 1, "satisfaction": 1}
    
    def evaluate_csv(self, csv_path: str, output_path: str = None) -> pd.DataFrame:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        
        # Apply sample limit if specified
        if SAMPLE_LIMIT is not None and SAMPLE_LIMIT < len(df):
            df = df.head(SAMPLE_LIMIT)
            print(f"Limited to {SAMPLE_LIMIT} samples")
        
        if output_path is None:
            filename = CSV_FILENAME.replace('.csv', '')
            output_path = f"/Users/yujin/Desktop/cocktail_rec_agentrag/ablation_llm+rag/result/{filename}.csv"
        
        evaluation_results = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating answers"):
            final_answer = str(row['final_answer']) if pd.notna(row['final_answer']) else ""
            query = str(row.get('query_EN', '')) if pd.notna(row.get('query_EN')) else ""
            context = str(row.get('final_context', '')) if pd.notna(row.get('final_context')) else ""
            
            if not final_answer.strip():
                scores = {"persuasiveness": 1, "transparency": 1, "accuracy": 1, "satisfaction": 1}
            else:
                scores = self.evaluate_answer(final_answer, query, context)
            
            evaluation_results.append(scores)
            time.sleep(API_DELAY)
        
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