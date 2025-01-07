import pandas as pd
from sklearn.metrics import  classification_report
from glob import glob
import os
import numpy as np
from collections import defaultdict
from transformers import AutoProcessor

import spacy
nlp = spacy.load("en_core_web_sm")
nlp.remove_pipe("lemmatizer")
nlp.add_pipe("lemmatizer", config={"mode": "lookup"}).initialize()
processor=AutoProcessor.from_pretrained("HuggingFaceM4/idefics-80b-instruct")

def editDistance(h, r, sent_len=None):
    #Reference: https://github.com/zszyellow/WER-in-python/blob/master/wer.py
    '''
    This function is to calculate the edit distance of reference sentence and the hypothesis sentence.

    Main algorithm used is dynamic programming.

    Attributes: 
        r -> the list of words produced by splitting reference sentence.
        h -> the list of words produced by splitting hypothesis sentence.
    '''
    d = np.zeros((len(r)+1)*(len(h)+1), dtype=np.uint8).reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        d[i][0] = i
    for j in range(len(h)+1):
        d[0][j] = j
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitute = d[i-1][j-1]+1
                insert = d[i][j-1]+1 
                delete = d[i-1][j]
                d[i][j] = min(substitute, insert, delete)
    # d=d[len(r)][len(h)]/len(r)
    d=d[len(r)][len(h)]
    
    if len(r)!=0:
        wnr=d/len(r)
        
    else:
        wnr=1 

    return d, wnr


def get_msg_distance(sentences):
 
    lemmatized_sentences = []
    sentences_lemma=[]
    for sentence in sentences:
        sentence = sentence.strip().strip('.').lower()
        doc = nlp(sentence)
        
     
        filtered_sentence_lemma = []
        for token in doc:
            if token.pos_ in ["NOUN", "ADJ", "VERB", "ADV", 'PROPN','NUM','PRON','ADP']:
                token_lemma=token.lemma_
                filtered_sentence_lemma.append(token_lemma)
          
        lemmatized_sentences.append(' '.join(filtered_sentence_lemma))
        sentences_lemma.append(filtered_sentence_lemma)
    


    wnd, wnr=editDistance(sentences_lemma[1], sentences_lemma[0])



    return wnd, wnr


def get_speaker_metrics(exp_type, model_type, speaker_result_fp, msg_type, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    print('-'*20)
    print(speaker_result_fp)
    all_results=glob(speaker_result_fp)
    print(len(all_results))

    msg_len_data=[]
    msg_wnd_data=[]
    msg_wnr_data=[]


    for result in all_results:
        # print(result)
        utter_len=[]
        utter_wnd=[]
        utter_wnr=[]

        df=pd.read_csv(result, index_col=0)

        
        for utter in df[msg_type]:
            utter_len.append(len(processor.tokenizer.tokenize(utter)))

        
        for targetName in df.targetImg.unique():
            sub_df=df[df['targetImg']==targetName]
            Img_utter_wer=[]
            Img_utter_wnr=[]
            prev_utter=None
   
            for utter in sub_df[msg_type]:
                if prev_utter:
                    msg_wnd, msg_wnr=get_msg_distance([prev_utter, utter])
                    Img_utter_wer.append(msg_wnd)
                    Img_utter_wnr.append(msg_wnr)
                prev_utter=utter
                
            utter_wnd.append(Img_utter_wer)
            utter_wnr.append(Img_utter_wnr)


        
        msg_len_data.append(utter_len)
        msg_wnd_data.append(utter_wnd)
        msg_wnr_data.append(utter_wnr)


    msg_len_data=np.array(msg_len_data).reshape(-1, 6, 4).mean(axis=2)
    msg_wnd_data=np.array(msg_wnd_data).mean(axis=1)
    msg_wnr_data=np.array(msg_wnr_data).mean(axis=1)

    # np.save(f'{output_dir}/{exp_type}_{model_type}_msg_len_data.npy', msg_len_data)
    # np.save(f'{output_dir}/{exp_type}_{model_type}_msg_wnd_data.npy', msg_wnd_data)
    # np.save(f'{output_dir}/{exp_type}_{model_type}_msg_wnr_data.npy', msg_wnr_data)
    msg_len_avg=msg_len_data.mean(axis=0)
    msg_wnd_avg=msg_wnd_data.mean(axis=0)
    msg_wnr_avg=msg_wnr_data.mean(axis=0)

    print("Msg Avg Length: ", msg_len_avg)
    print("Msg Avg WND: ", msg_wnd_avg)
    print("Msg Avg WNR: ", msg_wnr_avg)
    

    







def get_accuracies(exp_type, model_type, result_fp, target_column,pred_column, output_dir, first_round=1, last_round=6):
    print(result_fp)
    all_fps=glob(result_fp)
    print(len(all_fps))
    all_data_logs=[]
    for fp in all_fps:
    
        df=pd.read_csv(fp)
        interaction_logs=defaultdict(list)
        for i in range(first_round-1, last_round):
            golds=[]
            preds=[]      
            golds.append(df[target_column][4*i:4*(i+1)])
            preds.append(df[pred_column][4*i:4*(i+1)])

            golds=pd.concat(golds)
            preds=pd.concat(preds)

            
            report=classification_report(golds, preds, zero_division=0, output_dict=True)
         
           

            data_to_log={'rep_acc':report['accuracy']}
        
            for k, v in report.items():
                if k=='accuracy':
                    continue
                for kk, vv in v.items():
                    data_to_log[k+'_'+kk]=vv
            
            for k, v in data_to_log.items():
                interaction_logs[k].append(v)
        all_data_logs.append(interaction_logs)

   
    df=pd.DataFrame(all_data_logs)
    # df.to_csv(f'{output_dir}/accuracies_{exp_type}_{model_type}.csv')  
    accuracies=np.array(list(df['rep_acc']), dtype=float)
    mean_accuracy=accuracies.mean(axis=0)
    print(mean_accuracy)

def get_mtom_metrics(records_df):
    # Compare refined messages/predictions with the original
    refined_msgs = records_df["refined_msg"]
    original_msgs = records_df["gen_msg"]

    mtom_improvement = [
        editDistance(refined, original) for refined, original in zip(refined_msgs, original_msgs)
    ]

    print("MToM Improvement:", np.mean(mtom_improvement))



exp_type='explicitconsistency'
model_type='Claude' 
output_dir='evaluation_results'
pred_column='lsnr_pred'
target_column='tgt_label_for_lsnr'
msg_type='spkr_msg'
fp="outputs_icca/ICCA_data/*/records_*.csv"

get_accuracies(exp_type, model_type, fp, target_column=target_column,pred_column=pred_column, output_dir=output_dir)
get_speaker_metrics(exp_type, model_type, fp, msg_type, output_dir)
