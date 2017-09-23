from conllu.parser import parse

data = open('id-ud-train.conllu', 'r').read()
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
        
        temp_feature = []
        temp_feature.append(form)
        temp_feature.append(word_before)
        temp_feature.append(word_after)
        temp_feature.append(postag_before)
        
        feature_data.append(temp_feature)
        target_data.append(data_parsed[i][j].get('upostag'))

print feature_data
print target_data

'''from sklearn.model_selection import train_test_split
train_data, test_data, train_target, test_target = train_test_split(feature_data, target_data, test_size = 0.2)

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(train_data, train_target)
prediction = clf.predict(test_data)

from sklearn.metrics import accuracy_score
print 'accuracy = ', accuracy_score(test_target, prediction)'''