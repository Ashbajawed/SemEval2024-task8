#conda activate  semEval3.10

# python subtaskA/experiments/transformer.py \
#    --train_file_path /home/ashba.sameed/NLP_ass2/SemEval2024-task8/jsons/SubtaskA/subtaskA_train_monolingual.jsonl \
#    --test_file_path /home/ashba.sameed/NLP_ass2/SemEval2024-task8/jsons/SubtaskA/subtaskA_dev_monolingual.jsonl \
#    --prediction_file_path test.json \
#    --subtask A --model 'xlm-roberta-base'

python3 subtaskA/experiments/baseline_with_CNN.py \
   --train_file_path /home/ashba.sameed/NLP_ass2/SemEval2024-task8/jsons/SubtaskA/subtaskA_train_monolingual.jsonl \
   --test_file_path /home/ashba.sameed/NLP_ass2/SemEval2024-task8/jsons/SubtaskA/subtaskA_dev_monolingual.jsonl \
   --prediction_file_path pred_files/baseline_with_CNN.json \
   --subtask A --model 'xlm-roberta-base'


# python3 subtaskA/experiments/baseline_with_CRF.py \
#    --train_file_path /home/ashba.sameed/NLP_ass2/SemEval2024-task8/jsons/SubtaskA/subtaskA_train_monolingual.jsonl \
#    --test_file_path /home/ashba.sameed/NLP_ass2/SemEval2024-task8/jsons/SubtaskA/subtaskA_dev_monolingual.jsonl \
#    --prediction_file_path pred_files/baseline_with_CRF2.json \
#    --subtask A --model 'xlm-roberta-base'



python subtaskA/scorer/scorer.py \
--gold_file_path=/home/ashba.sameed/NLP_ass2/SemEval2024-task8/jsons/SubtaskA/subtaskA_dev_monolingual.jsonl \
--pred_file_path=pred_files/baseline_with_CNN.json
