import argparse
from src.himalaya_ridge import get_grouped_ridge_scores

targets = ['spatial expanse', 'interagent distance', 'agents facing', 'communication', 'joint action']

parser = argparse.ArgumentParser(description="Run encoding by task ID.")
parser.add_argument("--task_id", type=int, required=True, help="Task ID")
parser.add_argument("--max_tasks", type=int, required=True, help="Maximum number of tasks")
args = parser.parse_args()
task_id = args.task_id
max_tasks = args.max_tasks

# Use grouped ridge to combine each pose feature with the top perforrming layer (selected from cv from above row). 
# Use the combined feature set to predict each target. Tasks are automatically chunked and each use at most 4 CPUs
# Results are saved under experiments/ridge_results
# Each task takes about 10 - 30 minutes (usually around 20 minutes) with 4 cpus
get_grouped_ridge_scores(targets=targets, pose_features=['3D social pose features'], overwrite=False, task_id=task_id, max_tasks=max_tasks)
