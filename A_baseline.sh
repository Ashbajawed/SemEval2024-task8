python subtaskA/baseline/transformer_baseline.py \
   --train_file_path /home/ashba.sameed/NLP_ass2/SemEval2024-task8/jsons/SubtaskA/subtaskA_train_monolingual.jsonl \
   --test_file_path /home/ashba.sameed/NLP_ass2/SemEval2024-task8/jsons/SubtaskA/subtaskA_dev_monolingual.jsonl \
   --prediction_file_path baseline_A.json \
   --subtask A --model 'xlm-roberta-base'


python subtaskA/scorer/scorer.py \
--gold_file_path=/home/ashba.sameed/NLP_ass2/SemEval2024-task8/jsons/SubtaskA/subtaskA_dev_monolingual.jsonl \
--pred_file_path=baseline_A.json 
