from src.encoding import get_4dhumans_top_scores, get_4dhumans_encoding_scores


pose_features = [
    '2D social pose features', 
    '3D social pose features',
    # '3D body joints',
    # '3D vertices'
]
targets = ['spatial expanse', 'interagent distance', 'agents facing', 'communication', 'joint action']


# Collect the saved dnn encoding results from experiments/SOTA_beh
# Collect 4D Humans scores
get_4dhumans_encoding_scores(targets=targets, pose_features=pose_features, overwrite=False)
pose_model_score = get_4dhumans_top_scores(targets=targets, top_n=1, collect='target')
for index, row in pose_model_score.iterrows():
    print(index, row.to_dict())
# df_descriptive(pose_model_score)
