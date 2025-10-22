# 📊 Evaluation

It contains evaluation results and analysis files for the cocktail recommendation system.

## 📁 Directory Structure

```
evaluation/
├── final/                  # Final Evaluation Results File
├── ablation_wo_ref/        # Reflection Removal Ablation Study Results
├── initial_result/         # Initial evaluation results (including data with no search results)
├── final_eval.ipynb        # Final Results Analysis Notebook
├── llm_as_a_judge.py       # LLM-based evaluation script
└── readme.md               
```

## 🗂️ Evaluation File Structure

### Full System vs Ablation Study

The following table shows the location of the result file for each evaluation setting:

| Settings | GPT-4o-mini | GPT-5 |
|------|-------------|-------|
| **Full System** | `final/gpt-4o-mini_evaluation_v1.csv` | `final/gpt-4o-mini_evaluation_v2.csv` |
| **w/o Graph** | `final/gpt-4o-mini_wo_graph_v1.csv` | `final/gpt-4o-mini_wo_graph_v2.csv` |
| **w/o Reflection** | `ablation_wo_ref/gpt-4o-mini_evaluation_v1.csv` | `ablation_wo_ref/gpt-4o-mini_evaluation_v2.csv` |
| **w/o Graph & Reflection** | `ablation_wo_ref/gpt-4o-mini_wo_graph_v1.csv` | `ablation_wo_ref/gpt-4o-mini_wo_graph_v2.csv` |


### Python Dictionary Format

```python
evaluation_results = {
    # GPT-4o-mini as a judge
    'Full System (GPT-4o-mini)': './final/gpt-4o-mini_evaluation_v1.csv',
    'w/o graph (GPT-4o-mini)': './final/gpt-4o-mini_wo_graph_v1.csv',
    'w/o reflection (GPT-4o-mini)': './ablation_wo_ref/gpt-4o-mini_evaluation_v1.csv',
    'w/o graph w/o reflection (GPT-4o-mini)': './ablation_wo_ref/gpt-4o-mini_wo_graph_v1.csv',
    
    # GPT-5 as a judge
    'Full System (GPT-5)': './final/gpt-4o-mini_evaluation_v2.csv',
    'w/o graph (GPT-5)': './final/gpt-4o-mini_wo_graph_v2.csv',
    'w/o reflection (GPT-5)': './ablation_wo_ref/gpt-4o-mini_evaluation_v2.csv',
    'w/o graph w/o reflection (GPT-5)': './ablation_wo_ref/gpt-4o-mini_wo_graph_v2.csv',
}
```

## 📂 Folder Descriptions

### `initial_result/`
- Initial evaluation results file containing data with no search results
- Preservation of results from the initial experimental phase

### `final/`
- Result files used for the final evaluation
- Including Full System and w/o Graph variants

### `ablation_wo_ref/`
- Ablation Study Results for Removing the Reflection Module
- Including Additional Variations Based on Graph Presence

## 📓 Key Files

### `final_eval.ipynb`
- Final Evaluation Results Analysis and Visualization Notebook
- Includes performance comparisons and statistical analysis

### `llm_as_a_judge.py`
- Automated evaluation script utilizing LLM

---

**Note**: Evaluation metrics and detailed results can be found in `final_eval.ipynb`.
