#!/bin/sh
RUN_TYPE="$1"
RUN_NAME="$2"
SEED="$3"
if [ "$#" -eq 3 ]
    then
    tensorflowjs_converter --input_format=tf_saved_model --output_node_names='ppo_agent/ppo2_model/action_probs' --saved_model_tags=serve human_aware_rl/data/$RUN_TYPE/$RUN_NAME/seed$SEED/best human_aware_rl/data/web_models/$RUN_NAME

    cd tfjs-converter

    yarn ts-node tools/pb2json_converter.ts ../human_aware_rl/data/web_models/$RUN_NAME ../human_aware_rl/data/web_models/$RUN_NAME
else
echo "Should have 3 arguments: RUN_TYPE (ppo_runs/pbt_runs), RUN_NAME, and SEED"
exit 1
fi