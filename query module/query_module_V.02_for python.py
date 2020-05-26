
# coding: utf-8

# In[1]:


import json
from collections import Counter
import re, string 
import csv
import nltk
from nltk.tokenize import word_tokenize
import spacy
import string
import collections
import re
import copy
from pathlib import Path
import random
import pprint
from tqdm import tqdm_notebook


# In[4]:


def changepartsent(word, reword, sentence):
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
    
    if type(word) is list:
        for w in word:
            sentence = sentence.replace(w,reword)

    else:
        sentence = sentence.replace(word,reword)
        sentence = sentence.replace(word[0].upper()+word[1:],reword)
    
    return sentence


# In[5]:


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


# In[6]:


def openfile(filename):
    f = open(filename, 'w', encoding='utf-8', newline='')
    wr = csv.writer(f)
    
    return f, wr


# In[7]:


def closefile(f):
    f.close()


# In[8]:


def answerNER(text, word):
    '''
    Tagging the NER type of answer
    
    @param text
         	 query string
    @param word
         	 answer of the query

    @return list of NER
    '''
    doc = nlp(text)
    
    NERlist = []
    
    for ent in doc.ents:
        if set(ent.text.lower().split()) <= set(word.lower().split()) or set(word.lower().split()) <= set(ent.text.lower().split()): NERlist.append(ent.label_)
        #print(set(ent.text.lower().split()))
        #print(set(word.lower().split()))
        
        #print(set(ent.text.lower().split()) <= set(word.lower().split()))
        #print(set(word.lower().split()) <= set(ent.text.lower().split()))
        
        #print("--")
        
        #print((ent.text, ent.label_))
        #print("Q@#%$$@$#^$%")
    
    if(len(NERlist) == 0): NERlist.append('OTHER')
    
    return NERlist


# In[9]:


def findWHword(sentence):
    
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


# In[10]:


def ansTypeAnalysis(Data, mode):
    
    '''
    Find the answer type
    
    @param Data
         	 dictionary of triviaQA json file
    @param mode
         	 train or dev
             dev additionally process the reformulated question

    @return information about answertype
    '''
    
    anstype = {}
    for file in tqdm_notebook(Data):
        anstype[file['QuestionId']] = {}
        anstype[file['QuestionId']]['Question'] = file['Question']
        anstype[file['QuestionId']]['AnswerType'] = Counter()
        anstype[file['QuestionId']]['QueryType'] = []
        
        for can in candidate:
            s = file['Question'].lower()
            cleanedq = re.sub('[^0-9a-zA-Z]+', ' ', s)
            if can in cleanedq.split():
                if can == 'how' and 'how many' in s:
                    anstype[file['QuestionId']]['AnswerType']['how many'] += 1
                    anstype[file['QuestionId']]['QueryType'].append('counting')
                    continue
                anstype[file['QuestionId']]['AnswerType'][can] += 1
        
        if file['Question'].count('\"') > 0 or ((file['Question'].count('\'')-file['Question'].count('\'s')) % 2 == 0
                                               and (file['Question'].count('\'')-file['Question'].count('\'s')) > 0):
            anstype[file['QuestionId']]['QueryType'].append('quotes')
        
        if 'what' in anstype[file['QuestionId']]['AnswerType'].keys() or 'which' in anstype[file['QuestionId']]['AnswerType'].keys():
            da = detailAnsType(file['Question'])
            anstype[file['QuestionId']]['AnswerType_d'] = da[0]
            anstype[file['QuestionId']]['AnswerType_ds'] = da[1]
            
        if mode == 'train':
            anstype[file['QuestionId']]['Answer'] = file['Answer']['NormalizedValue']
           
            if anstype[file['QuestionId']]['AnswerType']['what'] or anstype[file['QuestionId']]['AnswerType']['which']:
                if anstype[file['QuestionId']]['AnswerType']['what']: wh = 'what'
                else: wh = 'which'
            
            else:
                for cand in ['when', 'how', 'where', 'who', 'how many']:
                    if anstype[file['QuestionId']]['AnswerType'][cand]:
                        wh = cand
                        break

            anstype[file['QuestionId']]['AnswerNER'] = answerNER(changepartsent(wh, file['Answer']['Value'], file['Question']), file['Answer']['Value'])
        
        if mode == 'dev':
            #anstype[file['QuestionId']]['Answer'] = file['Answer']['NormalizedValue']
           
            if anstype[file['QuestionId']]['AnswerType']['what'] or anstype[file['QuestionId']]['AnswerType']['which']:
                if anstype[file['QuestionId']]['AnswerType']['what']: wh = 'what'
                else: wh = 'which'
            
            else:
                for cand in ['when', 'how', 'where', 'who', 'how many']:
                    if anstype[file['QuestionId']]['AnswerType'][cand]:
                        wh = cand
                        break
                        
            anstype[file['QuestionId']]['reformedq'] = [changepartsent(wh, 'wildcard', file['Question']).replace("?","")]
            if 'quotes' in anstype[file['QuestionId']]['QueryType']:
                if file['Question'].count('\"') > 0:
                    anstype[file['QuestionId']]['reformedq'].append(changepartsent(re.findall(r'"([^"]*)"', anstype[file['QuestionId']]['reformedq'][0]), 'QUOTES', anstype[file['QuestionId']]['reformedq'][0]))
                    anstype[file['QuestionId']]['reformedq'].extend(re.findall(r'"([^"]*)"', anstype[file['QuestionId']]['reformedq'][0]))
                if (file['Question'].count('\'')-file['Question'].count('\'s')) % 2 == 0 and file['Question'].count('\'')-file['Question'].count('\'s') > 0:
                    anstype[file['QuestionId']]['reformedq'].append(changepartsent(re.findall(r"'(.*?)'", anstype[file['QuestionId']]['reformedq'][0]), 'QUOTES', anstype[file['QuestionId']]['reformedq'][0]))
                    anstype[file['QuestionId']]['reformedq'].extend(re.findall(r"'(.*?)'", anstype[file['QuestionId']]['reformedq'][0]))
                                                                    

    return anstype


# In[11]:


def countMoreThanOne(anstype, datatype, mode = 'dev'):
    
    '''
    (For statistic purpose)
    Count the queries which has more than one WH word
    '''
    
    moreThanOne = 0
    f, wr = openfile('countMoreThanOne_'+datatype+'_'+mode+'.csv')
    for question in anstype.values():
        if len(question['AnswerType']) > 1:
            wr.writerow([question['Question']])
            moreThanOne += 1
    
    closefile(f)
    
    return moreThanOne


# In[12]:


def countzero(anstype,datatype):
    '''
    (For statistic purpose)
    Count the queries which has zero WH word
    '''
    f, wr = openfile('countzero_'+datatype+'.csv')
    zeros = 0
    for question in anstype.values():
        if len(question['AnswerType']) == 0:
            zeros += 1
            wr.writerow([question['Question']])
            
    closefile(f)
    
    return zeros


# In[13]:


def WHstatistics(anstype,datatype):
    '''
    (For statistic purpose)
    Count the number of WH word type for entire query
    '''
    arrangedict = {}
    for question in anstype.values():
        arrangedict[question['Question']] = dict(question['AnswerType'])
        if 'AnswerType_d' in question:
            arrangedict[question['Question']]['AnswerType_d'] = question['AnswerType_d']

    #print(arrangedict)
    
    with open('WHstatistics_'+datatype+'.csv', "w") as f:
        header = ['Question'] + candidate + ['AnswerType_d']
        w = csv.DictWriter( f, header, lineterminator='\n' )
        w.writeheader()
        for key, val in arrangedict.items():
            row = {'Question': key}
            row.update(dict(val))
            w.writerow(row)          
            
            
    whcounter = Counter()
    for question in anstype.values():
        whcounter += question['AnswerType']
    
    f.close()
    
    return whcounter


# In[14]:


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


# In[15]:


def detailAnsType(question):
    '''
    Find detail answer type
    
    @return the detail answer type
    '''
    st = False
    ans = ''
    PTs = PosTagger(question)
    
#    if question == "What is the name of the world\'s largest church, that was begun in 1450, finished in 1600 and consecrated by Pope Urban XIII in 1626?":
#        print (PTs)
    
    for idx, w in enumerate(PTs):
        if w[0].lower() == 'what' or w[0].lower() == 'which':
            st = True
            continue
        if st == True and (w[1] == 'NN' or w[1] == 'NNP' or w[1] == 'NNS'):
            if 'kind' in w[0].lower() or 'name' in w[0].lower() or 'type' in w[0].lower() or (idx < len(PTs)-1 and "\'s" in PTs[idx+1][0].lower()):
                continue
                
            for j in range(idx, len(PTs)):
                if not (PTs[j][1] == 'NN' or PTs[j][1] == 'NNP' or PTs[j][1] == 'NNS'):
                    return (ans, PTs[j-1][0])
                ans += PTs[j][0]+' '
    
    return "NONE"


# In[16]:


def anstypedict(Data):
    '''
    Making dictionary for count the number of same simple answertype in same NER answer type
    '''
    adict = {}    
    for file in Data.values():
        #print(file['Question'])
        for ner in file['AnswerNER']:
            if ner not in adict: adict[ner] = Counter()
            if 'AnswerType_ds' in file:
                adict[ner][file['AnswerType_ds']] += 1
    
    return adict


# In[17]:


def sampleddata(n, Data, tp):
    filename = 'sampled_data'+tp+'.txt'
    filename2 = 'sampled_data_qid_'+tp+str(n)+'.txt'
    my_file = Path(filename)
    temp = []
    if my_file.is_file():
        f = open(filename, 'r')
        f2 = open(filename2,'w')
        while True:
            line = f.readline()
            if not line: break
            #print(line)
            #print(type(line))
            #print(len(Data))
            #print(int(line[:-1]))
            temp.append(Data[int(line[:-1])])
            f2.write(Data[int(line[:-1])]['QuestionId']+'\n')
        f.close()
        f2.close()
        
    else:
        index = random.sample(range(0, len(Data)), 1000)
        temp = [Data[i] for i in index]
        with open(filename, "w") as f:
            for t in index:
                f.write(str(t)+'\n')
        f.close()
    
    return temp


# In[ ]:


def main():
    parser = argparse.ArgumentParser("Preprocess Query Dataset")
    parser.add_argument("--train_file", default=config.TRIVIAQA_TRAIN)
    parser.add_argument("--dev_file", default=config.TRIVIAQA_DEV)
    parser.add_argument("--data_name", default="")
    
    print("Load training/dev dataset")
    with open(args.train_file, 'rt', encoding='UTF8') as f:
        trainwiki = json.load(f)
    
    with open(args.dev_file, 'rt', encoding='UTF8') as f:
        verifiedwiki = json.load(f)

    for file in verifiedwiki['Data']:
        file['Question'] = clearquestion(file['Question'])
        
    if not exists(config.CORPUS_DIR):
        mkdir(config.CORPUS_DIR)

    print("TEST file Analysis")
    print ("--Analysing answer type--")
    veriwikiansdev = ansTypeAnalysis(verifiedwiki['Data'],'dev')
    
    print("--Ratio more than one WH word in query--")
    print("wiki: ",countMoreThanOne(veriwikiansdev, 'wiki'),"/",len(verifiedwiki['Data']))

    print ("--Ratio zero WH word in query--")
    print("wiki: ", countzero(veriwikiansdev, 'wiki'),"/",len(verifiedwiki['Data']))

    print("--Create WHstatistics file--")
    WHstatistics(veriwikiansdev,'wiki')
    
    print("Train file Analysis")
    print ("--train anstype--")
    veriwikianstrain = ansTypeAnalysis(trainwiki['Data'],'dev')
    
    print("--Ratio more than one WH word in query--")
    print("wiki: ",countMoreThanOne(veriwikianstrain, 'wiki'),"/",len(verifiedwiki['Data']))

    print ("--Ratio zero WH word in query--")
    print("wiki: ", countzero(veriwikianstrain, 'wiki'),"/",len(verifiedwiki['Data']))

    print("--Create WHstatistics file--")
    WHstatistics(veriwikianstrain,'wiki')

    
    verifiedwiki_ntrain = copy.deepcopy(trainwiki)
    verifiedwiki_ndev = copy.deepcopy(verifiedwiki)
    
    print("Add field in dictionary")
    for q in verifiedwiki_ntrain['Data']:
        q['WHtype'] = veriwikianstrain[q['QuestionId']]['AnswerType']
        q['QueryType'] = veriwikianstrain[q['QuestionId']]['QueryType']
        if len(q['QueryType']) == 0: q['QueryType'].append('None')
        if 'AnswerType_d' in veriwikianstrain[q['QuestionId']]:
            q['AnswerType_d'] = veriwikianstrain[q['QuestionId']]['AnswerType_d']
            q['AnswerType_ds'] = veriwikianstrain[q['QuestionId']]['AnswerType_ds']
        q['reformedq'] = veriwikianstrain[q['QuestionId']]['reformedq']
        
    for q in verifiedwiki_ndev['Data']:
        q['WHtype'] = veriwikiansdev[q['QuestionId']]['AnswerType']
        q['QueryType'] = veriwikiansdev[q['QuestionId']]['QueryType']
        if len(q['QueryType']) == 0: q['QueryType'].append('None')
        if 'AnswerType_d' in veriwikiansdev[q['QuestionId']]:
            q['AnswerType_d'] = veriwikiansdev[q['QuestionId']]['AnswerType_d']
            q['AnswerType_ds'] = veriwikiansdev[q['QuestionId']]['AnswerType_ds']
        q['reformedq'] = veriwikiansdev[q['QuestionId']]['reformedq']
    
    print("Saving...")
    with open('./output/data_wiki_train.json', 'w') as outfile:
        json.dump(verifiedwiki_ntrain, outfile)
    with open('./output/data_wiki_dev.json', 'w') as outfile:
        json.dump(verifiedwiki_ndev, outfile)
    print("Done")


if __name__ == "__main__":
    candidate = ['when', 'how', 'where', 'which', 'what', 'who', 'how many', 'whose', 'whom']

    print("Load Spacy data")
    nlp = spacy.load('en_core_web_sm')
    main()

