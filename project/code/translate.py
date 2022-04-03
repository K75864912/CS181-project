from pygtrans import Translate
import json

from pygtrans.Null import Null 

with open("train.json", 'r', encoding="UTF-8") as f:
    train_data = json.load(f)
    
with open("test.json", 'r', encoding="UTF-8") as f:
    test_data = json.load(f)

with open("train_translate.json", 'a') as  f:
    # f.write('[')
    
    client = Translate()
    translate_train_data=[] # list
    i=1363
    for data in train_data[1364:]:
        translate_data_dic={}   # dic
        translated_text=client.translate(data['content'], target='en')
        while translated_text == Null:
            translated_text=client.translate(data['content'], target='en')
        translate_data_dic['content']=translated_text.translatedText
        translate_data_dic['label']=data['label']
        json.dump(translate_data_dic, f)
        i+=1
        print(i)
    f.write(']')


with open("test_translate.json", 'w') as  f:
    f.write('[')
    translate_test_data=[]  # list
    i=0
    for data in test_data:
        translate_test_dic={}   # dic
        translated_text=client.translate(data['content'], target='en')
        while translated_text == Null:
            translated_text=client.translate(data['content'], target='en')
        translate_test_dic['content']=translated_text.translatedText
        translate_test_dic['label']=data['label']
        json.dump(translate_test_dic, f)
        i+=1
        print(i)
    f.write(']')
