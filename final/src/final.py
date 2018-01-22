import jieba
import numpy as np
from gensim.models import word2vec
import os
import argparse
from math import sqrt
from math import pow

train_file_path = "data/training_data"
test_path = "data/testing_data.csv"
jieba.set_dictionary('dict.txt.big.txt')
merge_sentence_num = 5
word_dimension = 128

stopwordset = set()
with open('stopwords.txt','r') as sw:
    for line in sw:
        stopwordset.add(line.strip('\n'))

def load_train_data():
    X_train = []
    X = []
    temp_line = ""
    data_file = os.listdir(train_file_path)
    for file in data_file:
        data = open(train_file_path+"/"+file, 'r')
        count = 0
        
        for line in data:
            line = line.rstrip('\n')
            count += 1
            if count%merge_sentence_num !=0:
                temp_line += line
            else:
                temp_line += line
                X_train.append(list(jieba.cut(temp_line)))
                temp_line = ""
        '''
        for line in data:
            line = line.rstrip('\n')
            X.append(line)
        i = 0
        while i+merge_sentence_num < len(X):
            temp_line = ""
            for j in range(merge_sentence_num):
                temp_line += X[i+j]
            X_train.append(list(jieba.cut(temp_line)))
        '''
    '''
    # add stopword 
    for idx, line in enumerate(X_train):
        i = 0
        while i < len(line):
            if line[i] in stopwordset:
                del X_train[idx][i]
            else:
                i += 1
    # ================================
    '''
    return X_train
    
def load_testing_data():
    data = open(test_path, 'r')
    dialogue = []
    dialogue_origin = []
    question = []
    for line in data:
        line = line.replace("\n","")
        line = line.replace(":","")
        #line = line.replace("A:","")
        #line = line.replace("B:","")
        #line = line.replace("C:","")
        #line = line.replace("D:","")
        line = line.split(',')
        del line[0]
        line[0] = line[0].replace(" ","")
        line[0] = line[0].replace("\t","")
        line[1] = line[1].replace(" ","")
        line[1] = line[1].split('\t')
        dialogue.append(list(jieba.cut(line[0])))
        dialogue_origin.append(line[0])
        choices = []
        for opts in line[1]:
            choices.append(list(jieba.cut(opts)))
        question.append(choices)
    del dialogue[0] , question[0], dialogue_origin[0]
    print()
    return dialogue, question, dialogue_origin
    
def GetKeyWords(text,num):
    tr4w = TextRank4Keyword()
    tr4w.analyze(text=text, lower=True, window=2)   # py2中text必须是utf8编码的str或者unicode对象，py3中必须是utf8编码的bytes或者str对象
    return tr4w.get_keywords(num, word_min_len=1)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', action="store_true", default=False)
    parser.add_argument('-test', action="store_true", default=False)
    args = parser.parse_args()
    
    if args.train:
        print("Start train...")
        X_train = load_train_data()
        model = word2vec.Word2Vec(X_train, min_count=0, size=word_dimension, sg=1, window=50)
        model.save("word2vec.bin")
        
        
    elif args.test:
        print("Start testing...")
        model = word2vec.Word2Vec.load("word2vec.bin")
        dialogue, question, dialogue_origin = load_testing_data()
            
        
        dialogue_vec = [] 
        for d in dialogue:
            temp = np.zeros((word_dimension))
            count = 0
            for word in d:
                try:
                    temp += model.wv[word]
                    count += 1
                except:
                    pass
            dialogue_vec.append(temp/float(count))
        
        for idx1, choices in enumerate(question):
            for idx2, choice in enumerate(choices):
                temp = np.zeros((word_dimension))
                count = 0
                for word in choice:
                    try:
                        temp += model.wv[word]
                        count += 1
                    except:
                        pass
                if count!=0:
                    question[idx1][idx2] = temp/float(count)
                if count==0:
                    question[idx1][idx2] = temp
        
        
        for idx1, choices in enumerate(question):
            for idx2, choice in enumerate(choices):
                question[idx1][idx2] = np.dot(dialogue_vec[idx1], choice)
                question[idx1][idx2] = question[idx1][idx2] / (sqrt(np.sum(dialogue_vec[idx1]**2)) * sqrt(np.sum(choice**2)))
        
        with open("prediction.csv", 'w') as f:
            f.write('id,ans\n')
            for i in range(len(question)):
                f.write(str(i+1)+","+str(np.argmax(question[i]))+"\n")

            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            

    
    
    
    
    
    
    