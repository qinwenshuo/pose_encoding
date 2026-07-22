import argparse
from src.encoding import get_sota_encoding_scores

pose_features = [
    '2D social pose features', 
    '3D social pose features',
    # '3D body joints',
]
targets = ['spatial expanse', 'interagent distance', 'agents facing', 'communication', 'joint action']

parser = argparse.ArgumentParser(description="Run encoding by task ID.")
parser.add_argument("--task_id", type=int, required=True, help="Task ID")
parser.add_argument("--max_tasks", type=int, required=True, help="Maximum number of tasks")
args = parser.parse_args()
task_id = args.task_id
max_tasks = args.max_tasks

# Encode all DNN models with cross validation and test. Encode each target rating and pose feature with every layer. 
# The function utilizes all the available CPUs and automatially chunk all tasks by task id and max task number. 
# Encoding results are saved under experiments/SOTA_beh
# with 48 cpus, each task takes about 5 - 30 minutes (usually around 10 minutes)
get_sota_encoding_scores(targets=targets, pose_features=pose_features, overwrite=True, task_id=task_id, max_tasks=max_tasks)

