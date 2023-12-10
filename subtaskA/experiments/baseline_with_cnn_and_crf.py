from datasets import Dataset
import pandas as pd
import evaluate
import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, AutoTokenizer, set_seed
import os
from sklearn.model_selection import train_test_split
from scipy.special import softmax
import argparse
import logging
import time
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
stopwords = stopwords.words('english')
import time

def preprocess_function(examples, **fn_kwargs):
    return fn_kwargs['tokenizer'](examples["text"], truncation=True)
def lemmatize(text):
  lemmatizer = WordNetLemmatizer()
  r = re.sub('[^a-zA-Z]', ' ', text)
  r = r.lower()
  r= r.split()
  r = [word for word in r if word not in stopwords]
  r = [lemmatizer.lemmatize(word) for word in r]
  r = ' '.join(r)
  return r


def add_features(df):
  df['char_count'] = df['text'].apply(len) # Number of characters in the string
  df['word_count'] = df['text'].apply(lambda x: len(x.split())) # Number of words in the string 
  df['word_density'] = df['char_count'] / (df['word_count']+1) # Density of word (in char)
  df['title_word_count'] = df['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))
  df['text']= df['text'].apply(lemmatize)
  df['modified_word_count'] = df['text'].apply(lambda x: len(x.split())) # Number of words in the string 
  return df

def get_data(train_path, test_path, random_seed):
    """
    function to read dataframe with columns
    """
    train_df = pd.read_json(train_path, lines=True)
    test_df = pd.read_json(test_path, lines=True)
    train_df, test_df= add_features(train_df), add_features(test_df)
    
    train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df['label'], random_state=random_seed)

    return train_df, val_df, test_df

def compute_metrics(eval_pred):

    f1_metric = evaluate.load("f1")

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    results = {}
    results.update(f1_metric.compute(predictions=predictions, references = labels, average="micro"))

    return results


def fine_tune(train_df, valid_df, checkpoints_path, id2label, label2id, model):
    # pandas dataframe to huggingface Dataset
    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(valid_df)
    if os.path.exists(model):
        # If it exists, add a timestamp or a unique identifier
        timestamp = time.strftime("%Y%m%d%H%M%S")
        os.rename(model,  f"{model}_{timestamp}")
    # get tokenizer and model from huggingface
    tokenizer = AutoTokenizer.from_pretrained(model)     # put your model here
    model = AutoModelForSequenceClassification.from_pretrained(
       model, num_labels=len(label2id), id2label=id2label, label2id=label2id
    )    # put your model here
    
    # tokenize data for train/valid
    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True, fn_kwargs={'tokenizer': tokenizer})
    tokenized_valid_dataset = valid_dataset.map(preprocess_function, batched=True,  fn_kwargs={'tokenizer': tokenizer})
    

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


    # create Trainer 
    training_args = TrainingArguments(
        output_dir=checkpoints_path,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    # Define a custom CRF model
    class CRFModel(AutoModelForSequenceClassification):
        def __init__(self, config):
            super().__init__(config)
            self.num_labels = len(label2id)
            self.crf = CRF(num_tags=len(label2id))

        def forward(self, input_ids, attention_mask=None):
            outputs = super().forward(input_ids, attention_mask=attention_mask)
            logits = outputs[0]

            loss = self.crf(logits, attention_mask)
            return {"loss": loss}

    # Instantiate the CRF model
    model = CRFModel.from_pretrained(model)

    # create Trainer with CRF model
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_valid_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # save best model
    best_model_path = checkpoints_path+'/best/'
    
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)
    

    trainer.save_model(best_model_path)



def test(test_df, model_path, id2label, label2id):
    
    # load tokenizer from saved model 
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # load best model
    model = AutoModelForSequenceClassification.from_pretrained(
       model_path, num_labels=len(label2id), id2label=id2label, label2id=label2id
    )
            
    test_dataset = Dataset.from_pandas(test_df)

    tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True,  fn_kwargs={'tokenizer': tokenizer})
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # create Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    # get logits from predictions and evaluate results using classification report
    predictions = trainer.predict(tokenized_test_dataset)
    prob_pred = softmax(predictions.predictions, axis=-1)
    preds = np.argmax(predictions.predictions, axis=-1)
    metric = evaluate.load("bstrai/classification_report")
    results = metric.compute(predictions=preds, references=predictions.label_ids)
    
    # return dictionary of classification report
    return results, preds


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file_path", "-tr", required=True, help="Path to the train file.", type=str)
    parser.add_argument("--test_file_path", "-t", required=True, help="Path to the test file.", type=str)
    parser.add_argument("--subtask", "-sb", required=True, help="Subtask (A or B).", type=str, choices=['A', 'B'])
    parser.add_argument("--model", "-m", required=True, help="Transformer to train and test", type=str)
    parser.add_argument("--prediction_file_path", "-p", required=True, help="Path where to save the prediction file.", type=str)

    args = parser.parse_args()

    random_seed = 0
    train_path =  args.train_file_path # For example 'subtaskA_train_multilingual.jsonl'
    test_path =  args.test_file_path # For example 'subtaskA_test_multilingual.jsonl'
    model =  args.model # For example 'xlm-roberta-base'
    subtask =  args.subtask # For example 'A'
    prediction_path = args.prediction_file_path # For example subtaskB_predictions.jsonl

    if not os.path.exists(train_path):
        logging.error("File doesnt exists: {}".format(train_path))
        raise ValueError("File doesnt exists: {}".format(train_path))
    
    if not os.path.exists(test_path):
        logging.error("File doesnt exists: {}".format(train_path))
        raise ValueError("File doesnt exists: {}".format(train_path))
    

    if subtask == 'A':
        id2label = {0: "human", 1: "machine"}
        label2id = {"human": 0, "machine": 1}
    elif subtask == 'B':
        id2label = {0: 'human', 1: 'chatGPT', 2: 'cohere', 3: 'davinci', 4: 'bloomz', 5: 'dolly'}
        label2id = {'human': 0, 'chatGPT': 1,'cohere': 2, 'davinci': 3, 'bloomz': 4, 'dolly': 5}
    else:
        logging.error("Wrong subtask: {}. It should be A or B".format(train_path))
        raise ValueError("Wrong subtask: {}. It should be A or B".format(train_path))

    set_seed(random_seed)

    #get data for train/dev/test sets
    train_df, valid_df, test_df = get_data(train_path, test_path, random_seed)
    
    # train detector model
    PRED= prediction_path.split(".")[0].split("/")[-1]
    fine_tune(train_df, valid_df, f"checkpoints/{model}_{PRED}/subtask{subtask}/{random_seed}", id2label, label2id, model)

    # test detector model
    results, predictions = test(test_df, f"{model}/subtask{subtask}/{random_seed}/best/", id2label, label2id)
    
    logging.info(results)
    predictions_df = pd.DataFrame({'id': test_df['id'], 'label': predictions})
    predictions_df.to_json(prediction_path, lines=True, orient='records')
