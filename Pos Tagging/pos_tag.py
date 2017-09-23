from conllu.parser import parse

data = open('id-ud-train.conllu', 'r').read()
data_parsed = parse(data)

'''item = data_parsed[0][0]
print data_parsed[0][3].get('form')
print item.get('id'), "\n"
print item.get('form'), "\n"
print item.get('lemma'), "\n"
print item.get('upostag'), "\n"
print item.get('xpostag'), "\n"
print item.get('feats'), "\n"
print item.get('head'), "\n"
print item.get('deprel'), "\n"
print item.get('deps'), "\n"
print item.get('misc'), "\n"'''

list_data = []

for item in data_parsed:
    for inner_item in item:
        _form = str(inner_item.get('form'))
        _lemma = str(inner_item.get('lemma'))
        _upostag = str(inner_item.get('upostag'))
        _xpostag = str(inner_item.get('xpostag'))
        #_feats = str(inner_item.get('feats'))
        #_head = str(inner_item.get('head'))
        _deprel = str(inner_item.get('deprel'))
        _deps = str(inner_item.get('deps'))
        #_spaceafter = inner_item.get('misc').get('SpaceAfter')
        
        temp = []
        temp.append(_form)
        temp.append(_lemma)
        temp.append(_upostag)
        temp.append(_xpostag)
        #temp.append(_feats)
        #temp.append(_head)
        temp.append(_deprel)
        temp.append(_deps)
        #temp.append(_spaceafter)
        list_data.append(temp)

print list_data;