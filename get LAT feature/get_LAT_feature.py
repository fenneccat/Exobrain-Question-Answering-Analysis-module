#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python3
import datetime
import gc
import logging
import pickle
import os
import sys
import time, json

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import torch.utils.data as data_utils
from torch.autograd import Variable
from torch import optim

import data_utils
import models
from data_utils import to_torch
from eval_metric import mrr
from model_utils import get_pred_str, get_gold_pred_str, get_eval_string, get_output_index, store_predictions, store_probabilities
from tensorboardX import SummaryWriter

import csv
import numpy as np

from gensim.models import Word2Vec
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.models.wrappers import FastText

import spacy

from scipy import spatial
from tqdm import tqdm

import pandas as pd

sys.path.insert(0, './resources')
import config_parser, constant, eval_metric


# In[2]:


vocab = data_utils.get_vocab()


# In[3]:


vocab_set = vocab


# In[5]:


nlp = spacy.load('en')


# In[6]:


# In[ ]:


class TensorboardWriter:
  """
  Wraps a pair of ``SummaryWriter`` instances but is a no-op if they're ``None``.
  Allows Tensorboard logging without always checking for Nones first.
  """

  def __init__(self, train_log: SummaryWriter = None, validation_log: SummaryWriter = None) -> None:
    self._train_log = train_log
    self._validation_log = validation_log

  def add_train_scalar(self, name: str, value: float, global_step: int) -> None:
    if self._train_log is not None:
      self._train_log.add_scalar(name, value, global_step)

  def add_validation_scalar(self, name: str, value: float, global_step: int) -> None:
    if self._validation_log is not None:
      self._validation_log.add_scalar(name, value, global_step)


def get_data_gen(dataname, mode, args, vocab_set, goal, text, mention):
  dataset = data_utils.TypeDataset(constant.FILE_ROOT + dataname, lstm_type=args.lstm_type,
                                     goal=goal, vocab=vocab_set, text=text, mention=mention)
  if mode == 'train':
    data_gen = dataset.get_batch(args.batch_size, args.num_epoch, forever=False, eval_data=False,
                                 simple_mention=not args.enhanced_mention)
  elif mode == 'dev':
    data_gen = dataset.get_batch(args.eval_batch_size, 1, forever=True, eval_data=True,
                                 simple_mention=not args.enhanced_mention)
  else:
    data_gen = dataset.get_batch(args.eval_batch_size, 1, forever=False, eval_data=True,
                                 simple_mention=not args.enhanced_mention)
  return data_gen


def get_joint_datasets(args):
  train_gen_list = []
  valid_gen_list = []
  if args.mode == 'train':
    if not args.remove_open and not args.only_crowd:
      train_gen_list.append(
        #`("open", get_data_gen('train/open*.json', 'train', args, vocab, "open")))
        ("open", get_data_gen('distant_supervision/headwords.json', 'train', args, vocab, "open")))
      valid_gen_list.append(("open", get_data_gen('distant_supervision/headword_dev.json', 'dev', args, vocab, "open")))
    if not args.remove_el and not args.only_crowd:
      valid_gen_list.append(
        ("wiki",
         get_data_gen('distant_supervision/el_dev.json', 'dev', args, vocab, "wiki" if args.multitask else "open")))
      train_gen_list.append(
        ("wiki",
         get_data_gen('distant_supervision/el_train.json', 'train', args, vocab, "wiki" if args.multitask else "open")))
         #get_data_gen('train/el_train.json', 'train', args, vocab, "wiki" if args.multitask else "open")))
    if args.add_crowd or args.only_crowd:
      train_gen_list.append(
        ("open", get_data_gen('crowd/train_m.json', 'train', args, vocab, "open")))
  crowd_dev_gen = get_data_gen('crowd/dev.json', 'dev', args, vocab, "open")
  return train_gen_list, valid_gen_list, crowd_dev_gen


def get_datasets(data_lists, args, text, mention, vocab_set):
  data_gen_list = []
  for dataname, mode, goal in data_lists:
    data_gen_list.append(get_data_gen(dataname, mode, args, vocab_set, goal, text, mention))
  return data_gen_list


def _train(args):
  if args.data_setup == 'joint':
    train_gen_list, val_gen_list, crowd_dev_gen = get_joint_datasets(args)
  else:
    train_fname = args.train_data
    dev_fname = args.dev_data
    data_gens = get_datasets([(train_fname, 'train', args.goal),
                              (dev_fname, 'dev', args.goal)], args)
    train_gen_list = [(args.goal, data_gens[0])]
    val_gen_list = [(args.goal, data_gens[1])]
  train_log = SummaryWriter(os.path.join(constant.EXP_ROOT, args.model_id, "log", "train"))
  validation_log = SummaryWriter(os.path.join(constant.EXP_ROOT, args.model_id, "log", "validation"))
  tensorboard = TensorboardWriter(train_log, validation_log)

  model = models.Model(args, constant.ANSWER_NUM_DICT[args.goal])
  model.cuda()
  total_loss = 0
  batch_num = 0
  start_time = time.time()
  init_time = time.time()
  optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

  if args.load:
    load_model(args.reload_model_name, constant.EXP_ROOT, args.model_id, model, optimizer)

  for idx, m in enumerate(model.modules()):
    logging.info(str(idx) + '->' + str(m))

  while True:
    batch_num += 1  # single batch composed of all train signal passed by.
    for (type_name, data_gen) in train_gen_list:
      try:
        batch = next(data_gen)
        batch, _ = to_torch(batch)
      except StopIteration:
        logging.info(type_name + " finished at " + str(batch_num))
        torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                   '{0:s}/{1:s}.pt'.format(constant.EXP_ROOT, args.model_id))
        return
      optimizer.zero_grad()
      print(type_name)
      print("=================")
      print(batch)
      loss, output_logits = model(batch, type_name)
      loss.backward()
      total_loss += loss.data.cpu()[0]
      optimizer.step()

      if batch_num % args.log_period == 0 and batch_num > 0:
        gc.collect()
        cur_loss = float(1.0 * loss.data.cpu().clone()[0])
        elapsed = time.time() - start_time
        train_loss_str = ('|loss {0:3f} | at {1:d}step | @ {2:.2f} ms/batch'.format(cur_loss, batch_num,
                                                                                    elapsed * 1000 / args.log_period))
        start_time = time.time()
        print(train_loss_str)
        logging.info(train_loss_str)
        tensorboard.add_train_scalar('train_loss_' + type_name, cur_loss, batch_num)

      if batch_num % args.eval_period == 0 and batch_num > 0:
        output_index = get_output_index(output_logits)
        gold_pred_train = get_gold_pred_str(output_index, batch['y'].data.cpu().clone(), args.goal)
        accuracy = sum([set(y) == set(yp) for y, yp in gold_pred_train]) * 1.0 / len(gold_pred_train)
        train_acc_str = '{1:s} Train accuracy: {0:.1f}%'.format(accuracy * 100, type_name)
        print(train_acc_str)
        logging.info(train_acc_str)
        tensorboard.add_train_scalar('train_acc_' + type_name, accuracy, batch_num)
        for (val_type_name, val_data_gen) in val_gen_list:
          if val_type_name == type_name:
            eval_batch, _ = to_torch(next(val_data_gen))
            evaluate_batch(batch_num, eval_batch, model, tensorboard, val_type_name, args.goal)
    
    if batch_num % args.eval_period == 0 and batch_num > 0 and args.data_setup == 'joint':
      # Evaluate Loss on the Turk Dev dataset.
      print('---- eval at step {0:d} ---'.format(batch_num))
      feed_dict = next(crowd_dev_gen)
      eval_batch, _ = to_torch(feed_dict)
      crowd_eval_loss = evaluate_batch(batch_num, eval_batch, model, tensorboard, "open", args.goal)
    
    if batch_num % args.save_period == 0 and batch_num > 0:
      save_fname = '{0:s}/{1:s}_{2:d}.pt'.format(constant.EXP_ROOT, args.model_id, batch_num)
      torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, save_fname)
      print(
        'Total {0:.2f} minutes have passed, saving at {1:s} '.format((time.time() - init_time) / 60, save_fname))
  # Training finished! 
  torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
             '{0:s}/{1:s}.pt'.format(constant.EXP_ROOT, args.model_id))


def evaluate_batch(batch_num, eval_batch, model, tensorboard, val_type_name, goal):
  model.eval()
  loss, output_logits = model(eval_batch, val_type_name)
  output_index = get_output_index(output_logits)
  eval_loss = loss.data.cpu().clone()[0]
  eval_loss_str = 'Eval loss: {0:.7f} at step {1:d}'.format(eval_loss, batch_num)
  gold_pred = get_gold_pred_str(output_index, eval_batch['y'].data.cpu().clone(), goal)
  eval_accu = sum([set(y) == set(yp) for y, yp in gold_pred]) * 1.0 / len(gold_pred)
  tensorboard.add_validation_scalar('eval_acc_' + val_type_name, eval_accu, batch_num)
  tensorboard.add_validation_scalar('eval_loss_' + val_type_name, eval_loss, batch_num)
  eval_str = get_eval_string(gold_pred)
  print(val_type_name + ":" +eval_loss_str)
  print(gold_pred[:3])
  print(val_type_name+":"+ eval_str)
  logging.info(val_type_name + ":" + eval_loss_str)
  logging.info(val_type_name +":" +  eval_str)
  model.train()
  return eval_loss


def load_model(reload_model_name, save_dir, model_id, model, optimizer=None):
  if reload_model_name:
    model_file_name = '{0:s}/{1:s}.pt'.format(save_dir, reload_model_name)
  else:
    model_file_name = '{0:s}/{1:s}.pt'.format(save_dir, model_id)
  checkpoint = torch.load(model_file_name)
  model.load_state_dict(checkpoint['state_dict'])
  if optimizer:
    optimizer.load_state_dict(checkpoint['optimizer'])
  else:
    total_params = 0
    # Log params
    for k in checkpoint['state_dict']:
      elem = checkpoint['state_dict'][k]
      param_s = 1
      for size_dim in elem.size():
        param_s = size_dim * param_s
      print(k, elem.size())
      total_params += param_s
    param_str = ('Number of total parameters..{0:d}'.format(total_params))
    logging.info(param_str)
    print(param_str)
  logging.info("Loading old file from {0:s}".format(model_file_name))
  print('Loading model from ... {0:s}'.format(model_file_name))


    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(10332, 100)
        self.l2 = nn.Linear(100, 2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.l1(x))
        x = F.dropout(x, p = 0.5, training=self.training)
        x = F.log_softmax(self.l2(x), dim=1)
        return x

    # Load the answer classifier
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = Net()
    model.load_state_dict(checkpoint['state_dict'])
    return model

def eval(model, test_loader):
    scores = []
    for (cnt, data) in enumerate(test_loader):
        data = Variable(data, volatile=True).float()
        output = model(data)
        [a] = torch.exp(output).detach().numpy()
        scores.append(a)
    return scores

# return a list of the indexes where ch appears in s
def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]

def process_annot_id(annot_id):
    comma = find(annot_id, ",")
    paper_id = annot_id[1:comma[0]]
    mention = annot_id[comma[0]+1:comma[1]]
    score = annot_id[comma[1]+1:len(annot_id)-1] #score of the candidate answer
    return (paper_id, mention, score)

def _test(args):
  mention = "Deutsche Bundesbank balance of payments statistics"
  text = "We use the Deutsche Bundesbank balance of payments statistics as our main source of data."
  assert args.load
  test_fname = args.eval_data
  data_gens = get_datasets([(test_fname, 'test', args.goal)], args, text, mention)
  model = models.Model(args, constant.ANSWER_NUM_DICT[args.goal])
  model.cuda()
  model.eval()
  load_model(args.reload_model_name, constant.EXP_ROOT, args.model_id, model)
  
  for name, dataset in [(test_fname, data_gens[0])]:
     #print('Processing... ' + name)
     total_gold_pred = []
     total_probs = []
     total_ys = []
     total_annot_ids = []
     for batch_num, batch in enumerate(dataset):
       eval_batch, annot_ids = to_torch(batch)
       loss, output_logits = model(eval_batch, args.goal)
       output_index = get_output_index(output_logits)
       output_prob = model.sigmoid_fn(output_logits).data.cpu().clone().numpy()
       y = eval_batch['y'].data.cpu().clone().numpy()
       gold_pred = get_gold_pred_str(output_index, y, args.goal)
       total_probs.extend(output_prob)
       total_ys.extend(y)
       total_gold_pred.extend(gold_pred)
       total_annot_ids.extend(annot_ids)

    
def changepartsent(word, reword, sentence, lat):
    '''
    Change the part of sentence from word to reword
    @param word
         	@word that targeted to change
    @param reword
         	@word in sentence is replaced to @reword
    @param sentence
         	sentence string
    @return the changed string
    '''
    #print(word, reword, sentence)
    if type(word) is list:
        for w in word:
            sentence = sentence.replace(w,reword)

    else:
        sentence = sentence.replace(word,reword)
        sentence = sentence.replace(word[0].upper()+word[1:],reword)
        if lat != "":
            sentence = sentence.replace(lat, '')
    return sentence


def find_query(query_data, query_id):
    retrieved_query = {}
    for query in query_data['Data']:
        if query['QuestionId'] == query_id:
            retrieved_query = query
            break
    return retrieved_query

def similarity(w1, w2, fasttext_matrix):
    try:
        return fasttext_matrix.similarity(w1.lower(),w2.lower())
    except:
        return 0.0
    
def cosine_similarity(w1, w2, glove_dict):
    emb1 = data_utils.get_word_vec(w1, glove_dict)
    emb2 = data_utils.get_word_vec(w2, glove_dict)
    result = 1 - spatial.distance.cosine(emb1, emb2)
    return result
                   
def get_ultra_fine_entity_type(args, test_fname, model, vocab_set, mention, text):   
  data_gens = get_datasets([(test_fname, 'test', args.goal)], args, text, mention, vocab_set)

  for name, dataset in [(test_fname, data_gens[0])]:
     #print('Processing... ' + name)
     total_gold_pred = []
     total_probs = []
     total_ys = []
     total_annot_ids = []
     for batch_num, batch in enumerate(dataset):
       eval_batch, annot_ids = to_torch(batch)
       loss, output_logits = model(eval_batch, args.goal)
       output_index = get_output_index(output_logits)
       output_prob = model.sigmoid_fn(output_logits).data.cpu().clone().numpy()
       y = eval_batch['y'].data.cpu().clone().numpy()
       gold_pred = get_gold_pred_str(output_index, y, args.goal)
       total_probs.extend(output_prob)
       total_ys.extend(y)
       total_gold_pred.extend(gold_pred)
       total_annot_ids.extend(annot_ids)
  return gold_pred[0][1] #list of ultra fine entities

def doc_position(doc_list):
    doc_str = ""
    for doc in doc_list:
        doc_str += doc+'/'
    
    return doc_str

def entire_position(doc_list, pas_list):
    doc_pas_str = ""
    for doc in doc_list:
        for pas in pas_list:
            doc_pas_str += doc+'-'+pas+'/'
    
    return doc_pas_str

def get_doc_info(doc_dict):
    sen_pos_str = ""
#     for doc, pas_dict in doc_dict.items():
#         for pas, sent_list in pas_dict.items():
#             for sent in sent_list:
#                 sen_pos_str += doc+'-'+pas+'-'+sent+'/'
    
    for doc, pas_dict in doc_dict.items():
        for pas, sent_list in pas_dict.items():
            for sent in sent_list:
                sen_pos_str += doc+'-'+sent+'/'
    
    return sen_pos_str

def answer_type_similarity(args, test_fname, model,vocab_set):
    print("answer_type_similarity")
    with open("/home/haritz/projects/exobrain/data/data_wiki_dev.json", 'r') as query_file, open("/home/haritz/projects/exobrain/data/predictions_dev.json", 'r') as answer_file, open("/home/haritz/projects/exobrain/data/SNU_CGQA_dev_output_190807.csv", "w+") as csvfile:
        print("openeing queries data")
        queries_data = json.load(query_file)
        print("openeing answers data")
        answers_data_raw = json.load(answer_file)
        output_file = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        
        answers_data = dict()
        for instance in answers_data_raw:
            answers_data[instance['question_id']] = instance
            
        df_list = []
        
        count = 0
        for query_info in queries_data['Data']:
            if count % 100 == 0: print(count)
            count+=1
            
            wh_word = "" #let's take the wh word
            if "WHtype" in query_info and query_info['WHtype'] != {}:
                wh_word = next(iter(query_info['WHtype']))
            
            if wh_word.lower() != 'how many' and wh_word.lower() != 'how much' and wh_word != "":
                similarity = 0.0
                
                if query_info['QuestionId'] in answers_data:
                    candidates = answers_data[query_info['QuestionId']]['candidates'][:20]
                
                    for candidate in candidates: #for each answer
                        rem = []
                        max_sim = -float('inf')
                        #doc_p = doc_position(candidate['dids'])
                        #ent_p = entire_position(candidate['dids'], candidate['pids'])
                        doc_info = get_doc_info(candidate['doc_info'])
                        ans = candidate['symbol']
                        confidence = candidate['score']
                        lat = ""
                        if "AnswerType_d" in query_info:
                            lat = query_info['AnswerType_d']
                        ans_in_ctxt = changepartsent(wh_word, ans, query_info['Question'], lat)#Replace the wh word by the ans (we need the context to obtain the Name Entity)
                        ultra_fine_entity = get_ultra_fine_entity_type(args, test_fname, model, vocab_set, ans, ans_in_ctxt) 
                        #k1, k2 = get_ultra_fine_entity_type(args, test_fname, model, vocab_set, ans, ans_in_ctxt)
                        #ultra_fine_entity = get_ultra_fine_entity_type(args, test_fname, model, vocab_set, ans, "") 
                        for entity in ultra_fine_entity: #for each entity of the answer
                            if not "AnswerType_ds" in query_info: #if no lat, use WH word as answer type
                                similarity = cosine_similarity(entity, wh_word, vocab_set[1])
                                if similarity > max_sim:
                                    max_sim = similarity
                                    #rem = [query_info['QuestionId'], ans, confidence, similarity, doc_p, ent_p]
                                    rem = [query_info['QuestionId'], ans, confidence, similarity, doc_info]
                                #string = str(similarity) + " " + query_info['QuestionId'] + " " + query_info['Question'] + " " + ans + " " + entity + " " + wh_word + "\n"
                                #output_file.writerow([query_info['QuestionId'], ans, entity, wh_word, similarity, doc_p, ent_p])

                            else: #if there is lat
                                #simple lat
                                similarity = cosine_similarity(entity, query_info['AnswerType_ds'], vocab_set[1])
                                if similarity > max_sim:
                                    max_sim = similarity
                                    #rem = [query_info['QuestionId'], ans, confidence, similarity ,doc_p, ent_p]
                                    rem = [query_info['QuestionId'], ans, confidence, similarity, doc_info]
                                #string = str(sim) + " " + query_info['QuestionId'] + " " + query_info['Question'] + " " + ans + " " + entity + " " + query_info['AnswerType_ds']+ "\n"
                                #output_file.writerow([query_info['QuestionId'], ans, entity, query_info['AnswerType_ds'], similarity ,doc_p, ent_p])
                                #complex lat
                                for lat in query_info['AnswerType_d'].split():
                                    similarity = cosine_similarity(entity, lat, vocab_set[1])
                                    if similarity > max_sim:
                                        max_sim = similarity
                                        #rem = [query_info['QuestionId'], ans, confidence, similarity, doc_p, ent_p]
                                        rem = [query_info['QuestionId'], ans, confidence, similarity, doc_info]
                                    #string = str(sim) + " " + query_info['QuestionId'] + " " + query_info['Question'] + " " + ans + " " + entity + " " + lat+ "\n"
                                    #output_file.writerow([query_info['QuestionId'], ans, entity, lat, similarity, doc_p, ent_p])
                        #print(rem)
                        if rem != []:
                            #output_file.writerow(rem)
                            df_list.append(rem)
            
                    #ANSWER LEVEL
                    
                
            #QUESTION LEVEL
    
        return df_list


# In[7]:


def findWHword(sentence):
    
    candidate = ['when', 'how', 'where', 'which', 'what', 'who', 'how many', 'whose', 'whom']
    
    sentence = sentence.lower()
    
    if sentence.count('\"') > 0:
        sentence = changepartsent(re.findall(r'"([^"]*)"', sentence), 'QUOTES', sentence)
    if (sentence.count('\'')-sentence.count('\'s')) % 2 == 0 and sentence.count('\'')-sentence.count('\'s') > 0:
        sentence = changepartsent(re.findall(r"'(.*?)'", sentence), 'QUOTES', sentence)
        
        
    doc = nlp(sentence)
    
    if 'how' in sentence.split() and 'how many' in sentence:
        return 'how many'
    
    for w in reversed(doc):
        if w.pos_ == 'NN': continue
        else:
            for can in candidate:
                if can in w.text:
                    return can
            break
    
    whs = []

    for idx, token in enumerate(doc):
        for can in candidate:
            if can in token.text:
                return can
            
    if 'name' in sentence.lower() or doc[-1].lemma_ == 'be' or doc[-1].pos_ == 'ADP':
        return 'what'
    
    return 'None'


# In[8]:


def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)


# In[9]:


def detailAnsType(question):
    '''
    Find detail answer type
    
    @return the detail answer type
    '''
    st = False
    ans = ''
    PTs = PosTagger(question)
    
    #print (PTs)
    
    for idx, w in enumerate(PTs):
        if w[0].lower() == 'what' or w[0].lower() == 'which':
            st = True
            continue
        #print(w)
        if st == True and (w[1] == 'NN' or w[1] == 'NNP' or w[1] == 'NNS'):
            if 'kind' in w[0].lower() or 'name' in w[0].lower() or 'type' in w[0].lower() or (idx < len(PTs)-1 and "\'s" in PTs[idx+1][0].lower()):
                continue
                
            for j in range(idx, len(PTs)):
                if not (PTs[j][1] == 'NN' or PTs[j][1] == 'NNP' or PTs[j][1] == 'NNS'):
                    if "\'s" in PTs[j][0].lower():
                        ans = ''
                        break
                    else: return (ans, PTs[j-1][0])
                ans += PTs[j][0]+' '
                
    
    return ("None","None")


# In[10]:


def clearquestion(question):
    '''
    Clear the query if it contain unnecessary multiple quotes marks
    e.g) "What is the Milky Way?" --> What is the Milky Way?
    @param question
         	query string

    @return the cleaned setence
    '''
    
    if (question.startswith("\"") and question.endswith("\"")) or (question.startswith("\'") and question.endswith("\'")):
        question = question[1:-1]
        question = changepartsent("\"\"", "\"", question)
    
    return question


# In[16]:


def PosTagger(sentence):    
    '''
    POS Tagger
    
    @return the tuple of token and POS tag in sentence
    '''
    doc = nlp(sentence)
    poslist = []

    for token in doc:
        poslist.append((token.text, token.tag_))
    
    return poslist


# In[17]:


def extract_lat_feature(args, test_fname, model, vocab_set, query, pred):
    
    query = clearquestion(query)
    
    wh_word = findWHword(query)
    
    if wh_word.lower() == 'how many' or wh_word.lower() == 'how much':
        if hasNumbers(pred):
            similarity = 1.0
        else: similarity = 0.0
    
    elif wh_word == 'None':
        similarity = None
    
    else:
        similarity = 0.0
        
        max_sim = -float('inf')
        ans = pred
        
        lat_d, lat_s = detailAnsType(query)
        
        ans_in_ctxt = changepartsent(wh_word, ans, query, lat_d)#Replace the wh word by the ans (we need the context to obtain the Name Entity)
        ultra_fine_entity = get_ultra_fine_entity_type(args, test_fname, model, vocab_set, ans, ans_in_ctxt)
        
        for entity in ultra_fine_entity: #for each entity of the answer
            if lat_d == "None": #if no lat, use WH word as answer type
                similarity = cosine_similarity(entity, wh_word, vocab_set[1])
                if similarity > max_sim:
                    max_sim = similarity
                    
            else: #if there is lat
                #simple lat
                similarity = cosine_similarity(entity, lat_s, vocab_set[1])
                if similarity > max_sim:
                    max_sim = similarity
                    
                #complex lat
                for lat in lat_d.split():
                    similarity = cosine_similarity(entity, lat, vocab_set[1])
                    if similarity > max_sim:
                        max_sim = similarity
        
            similarity = max_sim
    
    return similarity


# In[ ]:


if __name__ == '__main__':
    config = config_parser.parser.parse_args()
    print("config: " + str(config))
    torch.cuda.manual_seed(config.seed)
    logging.basicConfig(
    filename=constant.EXP_ROOT +"/"+ config.model_id + datetime.datetime.now().strftime("_%m-%d_%H") + config.mode + '.txt',
    level=logging.INFO, format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M')
    logging.info(config)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    assert config.load
    test_fname = config.eval_data
    model = models.Model(config, constant.ANSWER_NUM_DICT[config.goal])
    model.cuda()
    model.eval()
    load_model(config.reload_model_name, constant.EXP_ROOT, config.model_id, model)
    
    query = None

    while query!= "":
        query = input("What is query? ")

        if query == "":
            break

        else:
            answer_candi = input("What is answer candidate? ")
            print("lat feature: {}".format(extract_lat_feature(config, test_fname, model,vocab_set, query, answer_candi)))

