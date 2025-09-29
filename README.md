# Cocktail Recommendation Agent RAG

Multi-Agent RAG system for cocktail recommendations.

## System Architecture

![Workflow](graph_viz/workflow.png)

The system operates through the following steps:

1. **user_question**: User input processing and preprocessing
2. **task_classification**: Classify question type into C1-C4 categories
3. **retriever**: Cocktail search based on classified task
4. **reflection**: Quality evaluation of search results (score-based)
5. **Conditional branching**:
   - **score < 80**: Additional search via `incremental_retriever` and re-evaluation
   - **score >= 80**: Final answer generation via `generator`

## Usage

### Running Jupyter Notebook

Use `user.ipynb` to test the system:

```python
# 1. Initialize RAG system
rag = RAG()

# 2. Text query
rag.query = "Please recommend a cocktail with refreshing mint"
result = rag.run()

# 3. Image + text combined query
rag.query = "Please recommend another cocktail using similar ingredients to this one"
rag.image_path = "image/mojito.jpeg"
result = rag.run()

# 4. Visualize results
display_comparison_results(result, rag.query, rag.image_path)
```

### Key Features

- **C1-C4 Task Classification**: Support for various cocktail recommendation scenarios
- **Multi-hop Search**: Ingredient-based expansion search (C3)
- **Image Analysis**: Cocktail recommendations through image analysis
- **Quality Evaluation**: Answer quality management through Reflection
- **Incremental Search**: Additional search when quality is low

## Environment Setup

```bash
# Required environment variables (.env file)
NEO4J_URI=neo4j://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
OPENAI_API_KEY=your_openai_api_key
```

## Getting Started

1. Set environment variables
2. Run Neo4j database
3. Execute `user.ipynb`
4. Input your cocktail questions

## Evaluation

1. **Evaluation data location**: `./data/cocktail_eval_data.csv`
2. **Run `eval_generate_{model_name}.py`**
   - Generated responses are saved as `{model_name}.csv` in `output/` directory
3. **Run `llm_as_judge.py`**
   - Evaluation results are saved as `{model_name}_evaluated.csv` in `result/` directory
   - Summary statistics are saved as `{model_name}_summary.csv` in `result/` directory

*Note: Files in `output/` and `result/` directories are examples for reference.*

## Experimental Results

We evaluated our system against a baseline approach using two evaluation models (GPT-4o-mini and GPT-5) across four key metrics:
![Evaluation](evaluation.png)


### Performance Comparison

| Evaluation Model | Metric | Ours | w/o graph | Improvement |
|------------------|--------|------|-----------|-------------|
| **GPT-4o-mini** | Persuasiveness | 2.98 | 2.83 | +0.15 (+5.30%) |
| | Transparency | 3.82 | 3.65 | +0.17 (+4.66%) |
| | Accuracy | 3.24 | 3.12 | +0.12 (+3.85%) |
| | Satisfaction | 3.00 | 2.83 | +0.17 (+6.01%) |
| | **Average** | **3.26** | **3.11** | **+0.15 (+4.82%)** |
| **GPT-5** | Persuasiveness | 2.00 | 1.92 | +0.08 (+4.17%) |
| | Transparency | 2.61 | 2.60 | +0.01 (+0.38%) |
| | Accuracy | 2.15 | 2.06 | +0.09 (+4.37%) |
| | Satisfaction | 1.94 | 1.87 | +0.07 (+3.74%) |
| | **Average** | **2.17** | **2.11** | **+0.06 (+2.84%)** |


*Evaluation methodology: LLM-as-a-Judge using 1-5 Likert scale ratings on 200 test cases per condition.*
