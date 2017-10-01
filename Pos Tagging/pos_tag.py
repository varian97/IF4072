from conllu.parser import parse
import sys
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
#import os
#os.chdir('Desktop\Informatika\Semester_7\IF4072_NLP\IF4072\Pos Tagging')

#data = open('id-ud-train.conllu', mode='r', encoding='utf-8').read() # For Python 3
data = open('id-ud-train.conllu', mode='r').read()
data_parsed = parse(data)

feature_data = []
target_data = []

for i in range(0, len(data_parsed)):
    for j in range(0, len(data_parsed[i])):
        # current word 
        form = data_parsed[i][j].get('form')
        
        # word before and its POS tag
        word_before = ""
        postag_before = ""
        if(data_parsed[i][j].get('id') == 1):
            word_before = "Null"
            postag_before = "Null"
        else:
            word_before = data_parsed[i][j-1].get('form')
            postag_before = data_parsed[i][j-1].get('upostag')
        
        # word after
        word_after = ""
        if(data_parsed[i][j].get('id') == len(data_parsed[i])):
            word_after = "Null"
        else:
            word_after = data_parsed[i][j+1].get('form')
        
        # features from http://nlpforhackers.io/training-pos-tagger/
        is_first = data_parsed[i][j].get('id') == 1
        is_last = data_parsed[i][j].get('id') == len(data_parsed[i])
        is_capitalized = form[0].upper() == form[0]
        prefix_2 = form[:2]
        prefix_3 = form[:3]
        suffix_2 = form[-2:]
        suffix_3 = form[-3:]
        is_numeric = form.isdigit()
        
        temp_feature = []
        temp_feature.append(form)
        temp_feature.append(word_before)
        temp_feature.append(word_after)
        temp_feature.append(postag_before)
        
        temp_feature.append(is_first)
        temp_feature.append(is_last)
        temp_feature.append(is_capitalized)
        temp_feature.append(prefix_2)
        temp_feature.append(prefix_3)
        temp_feature.append(suffix_2)
        temp_feature.append(suffix_3)
        temp_feature.append(is_numeric)
        
        feature_data.append(temp_feature)
        target_data.append(data_parsed[i][j].get('upostag'))

# PREPROCESSING STRING TO INTEGER (FACTORS)
import numpy as np
feature_data = np.array(feature_data)

from sklearn import preprocessing
column_preprocessor = []
for i in range(0, 12):
    _col_preprocessor = preprocessing.LabelEncoder()
    _col_preprocessor.fit(feature_data[:, i])
    feature_data[:, i] = _col_preprocessor.transform(feature_data[:, i])
    column_preprocessor.append(_col_preprocessor)

# Transforming target class using POS TAG Label Encoder (attribute 4)
target_data = column_preprocessor[3].transform(target_data)

# Save Label Encoder
for i in range(0, 12):
    np.save('encoders/encoder' + str(i) + '.npy', column_preprocessor[i].classes_)

################# USER INTERACTION ##########################################
clf = None

from sklearn import preprocessing
import numpy as np
POSTAG_ENCODER_INDEX = 3
label_encoders = []
for i in range(0, 12):
    encoder = preprocessing.LabelEncoder()
    encoder.classes_ = np.load('encoders/encoder' + str(i) + '.npy')
    label_encoders.append(encoder)
    
def features(sentence, index, postag_before):
    return [
        sentence[index],                                                #current_word
        "Null" if index == 0 else sentence[index - 1],                  #word_before
        "Null" if index == len(sentence) - 1 else sentence[index + 1],  #word_after
        "Null" if index == 0 else postag_before,                        #postag_before
        index == 0,                                                     #is_first
        index == len(sentence) - 1,                                     #is_last
        sentence[index][0].upper() == sentence[index][0],               #is_capitalized
        sentence[index][:2],                                            #prefix_2
        sentence[index][:3],                                            #prefix_3
        sentence[index][-2:],                                           #suffix_2
        sentence[index][-3:],                                           #suffix_3
        sentence[index].isdigit()                                       #is_numeric
    ]

def pos_tag(sentence, index, postag_before):
    feature = features(sentence, index, postag_before)
    for i in range(0, 12):
        feature[i] = label_encoders[i].transform([feature[i]])[0]
    return clf.predict([feature])
    
#def pos_tag(sentence):
#    tags = clf.predict([features(sentence, index) for index in range(len(sentence))])
#    return zip(sentence, tags)


while(True) :
    print("\n=========================")
    print("What do you want to do ?")
    print("=========================")
    print("1. Load Model")
    print("2. Train Model")
    print("3. Start Pos-Tagging")
    print("4. Exit")
    choice = input("Your choice : ")
    
    if(choice == '1'):
        try:
            filename = input("File name: ")
            filename = "model/" + filename
            clf = joblib.load(filename)
            print("Model Loaded !")
        except IOError:
            print("\nFile not found !")
            
    elif(choice == '2'):
        train_data, test_data, train_target, test_target = train_test_split(feature_data, target_data, test_size = 0.2)
    
        print("\nSelect the classifier :")
        print("1. Random Forest")
        print("2. Decision Tree Classifier")
        cls_choice = input("Your choice : ")
        
        if(cls_choice == '1'):
            print ("Training using Random Forest Classifier : ")
            clf = RandomForestClassifier(n_estimators=250)
            clf.fit(train_data, train_target)
            
            prediction = clf.predict(test_data)
            
            print ('Accuracy using Random Forest = ', accuracy_score(test_target, prediction))
        else:
            print ("Training using Decision Tree Classifier : ")
            clf = DecisionTreeClassifier()
            clf.fit(train_data, train_target)
            
            prediction = clf.predict(test_data)
            print ('\nAccuracy using DTL = ', accuracy_score(test_target, prediction))
            
        issave = input("\n\nYou want to save this model ? (y/n): ")
        if(issave.lower() == 'y'):
            filename = input("File name: ")
            filename = "model/" + filename
            joblib.dump(clf, filename)
            print("\nModel has been saved !")
            
    elif(choice == '3'):
        if(clf != None):
            sentence = input("Input the sentence : ")
            sentence = sentence.split(" ")
            
            current_tag = 'NULL'
            for i in range(0, len(sentence)):
                current_tag = pos_tag(sentence, i, current_tag)
                current_tag = label_encoders[POSTAG_ENCODER_INDEX].inverse_transform(current_tag)[0]
                print(sentence[i], '->', current_tag, '\n')
            
            # ERROR karena hasil label encoder cuman 1 dimensi
            # Method pos_tag sama feature, butuh sentence sebagai list of word
            
            #le = preprocessing.LabelEncoder()
            #le.fit(sentence)
            #sentence = le.transform(sentence)
            
            #print("Result : \n")
            #print(pos_tag(sentence))'''
        else:
            print("\nPlease train the model or load it !\n")
        
    else:
        sys.exit(0)