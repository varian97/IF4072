from conllu.parser import parse
import os
os.chdir('Desktop\Informatika\Semester_7\IF4072_NLP\IF4072\Pos Tagging')

data = open('id-ud-train.conllu', mode='r', encoding='utf-8').read()
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

target_data = column_preprocessor[3].transform(target_data)

# TRAINING MODEL
from sklearn.model_selection import train_test_split
train_data, test_data, train_target, test_target = train_test_split(feature_data, target_data, test_size = 0.2)

# Random Forest
print "Training using Random Forest Classifier : "
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=250)
clf.fit(train_data, train_target)
prediction = clf.predict(test_data)

from sklearn.metrics import accuracy_score
print ('Accuracy using Random Forest = ', accuracy_score(test_target, prediction))

print "============================================"

# DTL
print "Training using Decision Tree Classifier : "
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(train_data, train_target)
prediction = clf.predict(test_data)

from sklearn.metrics import accuracy_score
print ('Accuracy using DTL = ', accuracy_score(test_target, prediction))