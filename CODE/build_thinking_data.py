

# def build_data(his_txt,en,aes,surface,background,enem):
#     tem = "<think>Let's solve this problem step by step. In the first step, since we need to obtain all entities related to the events of the image and sentence, we first need to obtain the background events and historical information related to the image and sentence:"
#     his_txt=str(his_txt)
#     tem1="Secondly, we need to extract entity words from the historical background information. We need to first clarify the definition of entity words as follows:  An 'entity' can be a person, product, brand, location, event. Therefore, entity words can be extracted from historical background information:"
#     en=str(en)
#     tem2=". In the second step, since we need to judge the emotion of the entity and the reasons behind it from multiple different perspectives, we first need to obtain the surface information related to the image and sentence, that is, the intuitive visual experience of the image including the aesthetic information of the image:"
#     aes=str(aes)
#     tem3=", the facial expressions of the characters in the image, the scene information of the image, and the intuitive meaning of the text. Finally, from the perspective of intuitive and surface information, the emotions of each entity and the reasons for this emotion are analyzed as follows:"
#     surface=str(surface)
#     tem4="Secondly, in order to obtain emotions and the reasons for emotions from a deeper level, it is necessary to use the historical background information obtained in the previous step, from a deeper perspective and historical events, we can analyze the emotions of each entity and the reasons for their emotions:"
#     background=str(background)
#     tem5="Therefore, the final entities and emotions that can be obtained are:</think>"
#     enem=str(enem)
#     thinking_data = tem + his_txt + tem1+en+tem2+aes+tem3+surface+tem4+background+tem5+enem
#     return thinking_data

def build_data(his_txt,en,aes,surface,background,enem):
    tem = "<think>Let's solve this problem step by step. Firstly, we need to obtain the historical background information related to the image and sentence:"
    his_txt=str(his_txt)
    tem1=". Secondly, we extract from the historical background information the relevant entities:"
    en=str(en)
    tem2=". Thirdly, we obtain the surface information related to the image and sentence, that is, the intuitive visual experience of the image including the aesthetic information of the image:"
    aes=str(aes)
    tem3=", the facial expressions of the characters, the scene information of the image, and the intuitive meaning of the text. Next, we identify each entity's sentiment and the intuitive causes behind it:"
    surface=str(surface)
    tem4=". Finally, we utilize the historical background information from the previous step to analyze each entity's sentiment and their deeper perspective causes:"
    background=str(background)
    tem5=". Therefore, the final entities and sentiment that can be obtained are:</think>"
    enem=str(enem)
    thinking_data = tem + his_txt + tem1+en+tem2+aes+tem3+surface+tem4+background+tem5+enem
    return thinking_data

with open('/data2/liuxj/1-Sentiment-mllm/model_train/data/filter_data/enem_cleaned.txt','r') as fe:
    line = fe.readline()
    ens=[]
    enems=[]
    while line:
        id,enem = line.strip().split('\t')
        enem = eval(enem)
        en = [ei[0] for ei in enem]
        ens.append(en)
        enems.append(enem)
        line = fe.readline()

with open('/data2/liuxj/1-Sentiment-mllm/model_train/data/filter_data/his_and_es_cleaned.txt','r') as fe:
    line = fe.readline()
    hiss=[]
    while line:
        _,his,_ = line.strip().split('\t')
        hiss.append(his)
        line = fe.readline()

with open('/data2/liuxj/1-Sentiment-mllm/model_train/data/filter_data/Aesthetic_cleaned.txt','r') as fe:
    line = fe.readline()
    aess=[]
    while line:
        _,aes= line.strip().split('\t')
        aess.append(aes)
        line = fe.readline()

with open('/data2/liuxj/1-Sentiment-mllm/model_train/data/filter_data/surface_sum_cleaned.txt','r') as fe:
    line = fe.readline()
    surs=[]
    while line:
        _,sur = line.strip().split('\t')
        surs.append(sur)
        line = fe.readline()

with open('/data2/liuxj/1-Sentiment-mllm/model_train/data/filter_data/background_sum_cleaned.txt','r') as fe:
    line = fe.readline()
    backs=[]
    ids = []
    while line:
        id,back = line.strip().split('\t')
        backs.append(back)
        ids.append(id)
        line = fe.readline()

count=0
with open('data/filter_data/thinking_data.txt','a') as fa:
    for his, en ,aes ,surface,background,enem,id in zip(hiss,ens,aess,surs,backs,enems,ids):
        thinking_data = build_data(his, en ,aes ,surface,background,enem)
        fa.write(id)
        fa.write('\t')
        fa.write(thinking_data)
        fa.write('\n')
        count+=1
print(count)
