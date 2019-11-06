#!/bin/sh
RUN_TYPE="$1"
RUN_NAME="$2"
SEED="$3"
if [ "$#" -eq 3 ]
then
    if [ "$1" = "ppo_runs" ]
    then
        tensorflowjs_converter --input_format=tf_saved_model --output_node_names='ppo_agent/ppo2_model/action_probs' --saved_model_tags=serve human_aware_rl/data/$RUN_TYPE/$RUN_NAME/seed$SEED/ppo_agent \
            human_aware_rl/data/web_models/$RUN_NAME\_seed$SEED\_temp
    elif [ "$1" = "pbt_runs" ]
    then
        tensorflowjs_converter --input_format=tf_saved_model --output_node_names='agent0/ppo2_model/action_probs' --saved_model_tags=serve human_aware_rl/data/$RUN_TYPE/$RUN_NAME/seed_$SEED/agent0/best \
            human_aware_rl/data/web_models/$RUN_NAME\_seed$SEED\_temp
    else
        echo "Should have 3 arguments: RUN_TYPE (ppo_runs/pbt_runs), RUN_NAME, and SEED"
        exit 1
    fi
    
    cd tfjs-converter
    yarn ts-node tools/pb2json_converter.ts ../human_aware_rl/data/web_models/$RUN_NAME\_seed$SEED\_temp \
        ../human_aware_rl/data/web_models/$RUN_NAME\_seed$SEED
    

    rm -rf ../human_aware_rl/data/web_models/$RUN_NAME\_seed$SEED\_temp
else
    echo "Should have 3 arguments: RUN_TYPE (ppo_runs/pbt_runs), RUN_NAME, and SEED"
    exit 1
fi
