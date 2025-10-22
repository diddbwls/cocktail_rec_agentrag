import pandas as pd
import openai
import json
import os
import sys
from typing import Dict
import time
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from parent directory
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path, override=True)  # override=True to overwrite existing env vars

# Debug: Check which API key is loaded
api_key = os.getenv('OPENAI_API_KEY')
if api_key:
    print(f"✅ API Key loaded: {api_key[:20]}...{api_key[-4:]}")
else:
    print("❌ No API key found in environment")

# Add parent directory to path to import prompts
sys.path.append(str(Path(__file__).parent.parent))
from prompts.llm_judge_prompt import get_evaluation_prompt

# Configuration
# Target CSV filename to evaluate
CSV_FILENAME = "gpt-4o-mini_evaluation_v2.csv" 

# Number of samples to evaluate (None = all samples, or set a number like 10)
MAX_SAMPLES =200  # None means evaluate all data
MODEL_NAME = "gpt-5"

'''
Running this script will save evaluation results and summary files to the result folder
Evaluation uses standardized prompt from prompts/llm_judge_prompt.py
Scoring: 1-5 Likert scale (1=strongly disagree, 5=strongly agree)
--------------------------------
Output format:
Persuasiveness: score (±std_deviation)
Transparency: score (±std_deviation) 
Accuracy: score (±std_deviation)
Satisfaction: score (±std_deviation)
Overall Average: score (±std_deviation)
'''


class LLMAsJudge:
    def __init__(self, api_key: str = None):
        self.client = openai.OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
        
    def evaluate_answer(self, final_answer: str, query: str, context: str = "") -> Dict[str, int]:
        # Use standardized prompt from prompts/llm_judge_prompt.py
        prompt = get_evaluation_prompt(answer=final_answer, query=query, context=context)
        
        max_retries = 3
        retry_delay = 3
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=MODEL_NAME,  
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}  # Force JSON output format
                )
                
                result = json.loads(response.choices[0].message.content.strip())
                return result
                
            except openai.NotFoundError as e:
                print(f"⚠️  Model 'gpt-5' not found. Please check if you have access to GPT-5.")
                print(f"Error details: {e}")
                return {"persuasiveness": 0, "transparency": 0, "accuracy": 0, "satisfaction": 0}
            except openai.RateLimitError as e:
                if attempt < max_retries - 1:
                    print(f"Rate limit hit. Retrying in {retry_delay} seconds... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    print(f"Rate limit error after {max_retries} attempts: {e}")
                    return {"persuasiveness": 0, "transparency": 0, "accuracy": 0, "satisfaction": 0}
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON response: {e}")
                print(f"Response content: {response.choices[0].message.content if response else 'No response'}")
                return {"persuasiveness": 0, "transparency": 0, "accuracy": 0, "satisfaction": 0}
            except Exception as e:
                print(f"Error evaluating answer: {e}")
                return {"persuasiveness": 0, "transparency": 0, "accuracy": 0, "satisfaction": 0}
    
    def evaluate_csv(self, csv_path: str, output_path: str = None, max_samples: int = None) -> pd.DataFrame:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        
        # Limit number of samples if specified
        if max_samples is not None:
            df = df.head(max_samples)
            print(f"📊 Evaluating first {len(df)} samples (limited by MAX_SAMPLES={max_samples})")
        else:
            print(f"📊 Evaluating all {len(df)} samples")
        
        if output_path is None:
            filename = CSV_FILENAME.replace('.csv', '')
            output_path = f"./result_reflection없는llmjudge/{filename}.csv"
        
        evaluation_results = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating answers"):
            final_answer = str(row['round1_answer']) if pd.notna(row['round1_answer']) else ""
            query = str(row.get('query_EN', '')) if pd.notna(row.get('query_EN')) else ""
            context = str(row.get('round1_context', '')) if pd.notna(row.get('round1_context')) else ""
            
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
    


def main():
    csv_path = f"./output/{CSV_FILENAME}"
    judge = LLMAsJudge()
    
    # Start timing
    start_time = time.time()
    print(f"Starting evaluation of: {CSV_FILENAME}")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    
    try:
        evaluated_df = judge.evaluate_csv(csv_path, max_samples=MAX_SAMPLES)
        
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

