# Remote Sensing Evaluator

A remote sensing task evaluator built on the OpenAI Response API, used to assess
whether AI-generated code and its execution results satisfy task requirements.

## Running

Dependencies and configuration are shared with the root project. Install
dependencies with `pip install -r ../requirements.txt` and configure API keys in
the root `.env` (see [Environment Configuration](#environment-configuration)).
Set the evaluator input in `main.py`, then run it from this directory with
`cd Eval && python main.py`.

## Features

- **Multi-dimensional evaluation**: evaluates six dimensions — task consistency,
  technical correctness, result completeness, accuracy, visualization quality, and
  code quality
- **Code execution**: executes reference code and generated code in a sandbox
  environment
- **Multi-format support**: supports reading and analyzing remote sensing images,
  GeoTIFF, shapefiles, GeoJSON, and other file types
- **Comparative analysis**: compares generated results with the reference code and
  points out strengths and weaknesses

## Evaluation Dimensions

| Dimension              | Description                                                       |
| ---------------------- | ----------------------------------------------------------------- |
| Task consistency       | Whether the result addresses the task description and whether the output geographic extent is correct |
| Technical correctness  | Whether the file format, coordinate system, and spatial resolution are correct |
| Result completeness    | Whether all required outputs are included                          |
| Accuracy               | Whether boundary locations and class assignments are accurate      |
| Visualization quality  | Whether images are clear and readable with reasonable color schemes |
| Code quality           | Whether the code logic is clear and the differences from the reference code are reasonable |

## Project Structure

```
eval_agent/
├── eval/
│   └── agent.py          # Evaluator agent implementation
├── sandbox/              # Sandbox execution environment
├── dataset/              # Dataset
├── scratch/              # Temporary file directory
└── main.py               # Entry point
```

Dependencies (`requirements.txt`) and configuration (`.env`, `config.ini`) live
at the project root and are shared across the whole project.

## Environment Configuration

```bash
# Configure the root-level .env file (shared across the whole project)
OPENAI_API_KEY=your_api_key
OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_MODEL=qwen3.5-plus
```

## Usage

```bash
# Run the evaluation from the Eval directory
cd Eval && python main.py
```

Evaluator input format (`eval/test.json`):

```json
{
  "task_text": "task description",
  "reference_code": "reference code",
  "generated_code": "generated code"
}
```

## Output Format

Evaluation results are in JSON format, containing six dimension scores (0-1) and an
overall score:

```json
{
  "task_consistency": {"score": 0.0-1.0, "comments": "..."},
  "technical_correctness": {"score": 0.0-1.0, "comments": "..."},
  "result_completeness": {"score": 0.0-1.0, "comments": "..."},
  "accuracy": {"score": 0.0-1.0, "comments": "..."},
  "visualization_quality": {"score": 0.0-1.0, "comments": "..."},
  "code_quality": {"score": 0.0-1.0, "comments": "..."},
  "overall_score": 0.0-1.0,
  "summary": "overall evaluation",
  "suggestions": ["improvement suggestions"],
  "reference_vs_generated": {
    "strengths": ["strengths"],
    "weaknesses": ["weaknesses"],
    "differences": ["differences"]
  }
}
```
