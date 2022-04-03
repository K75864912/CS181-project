from langdetect import detect
import json 

with open("train.json", 'r', encoding="UTF-8") as f:
    train_data = json.load(f)
    
with open("test.json", 'r', encoding="UTF-8") as f:
    test_data = json.load(f)
   
# check how many languages in the training set 
language_dic={}
i=1
#for data in train_data:
for data in test_data:    
    language=detect(data['content'])
    print(i,end='')
    print(language)
    i+=1
    if(language_dic.get(language)):
        language_dic[language]+=1
    else:
        language_dic[language]=1
    
print(language_dic)