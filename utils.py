
import pandas as pd
from collections import Counter
import pyarabic.araby as araby
import gensim
from tashaphyne.stemming import ArabicLightStemmer

import tensorflow_hub as hub
from tensorflow import compat
import numpy as np
import ast 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity 
# from sentence_transformers import SentenceTransformer

from scipy.stats import pearsonr

import tensorflow as tf
# Imports
# from datasets import load_dataset
import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()

import json
ArListem = ArabicLightStemmer()

def clean(file_name):
    data = pd.read_csv(file_name)
    size = len(data['text'])
    stems=[]
    for aya_number in range(size):
        stems.append(preprocess_aya(data['text'][aya_number]))
    data['stem']=pd.Series(s for s in stems)
    data.to_csv('cleaned_stem.csv')



def preprocess_aya(doc:str)->list:
    """
    Preprocess the text by applying several NLP techniques:
    Tokenisation, removing diacritics, removing stop words, get stemmers/roots...
    """
    #Removing diactritics
    doc = araby.strip_diacritics(doc)
    #Tokenize + remove stop_words
    doc = set(araby.tokenize(doc)) - _get_quran_stopwords()
    #Stemming:loop
    aya_stemmed = [_get_stem(token) for token in doc]
    return aya_stemmed


def _get_quran_stopwords()->set:
    stop_words={}
    with open('stop_words.txt', 'r', encoding="utf-8") as f:
        data = f.read()
        stop_words = set(data.split(','))
    
    return stop_words


def _get_stem(word)->str:
    stem = ArListem .light_stem(word)
    return stem


def _get_root(word)->str:
    ArListem .light_stem(word)

    root = ArListem.get_root()
    return root


def words_to_text_embedding(words:list):
    print('*'*30)
    """
    calculate the mean of word vectors
    """
    text_embedding=[]
    text_embedding=np.mean(words, axis=0)
    print('#'*30)
    return text_embedding

        
def get_embedding_vector(doc)->list:
    tf=compat.v1
    tf.disable_eager_execution()
    if isinstance(doc,str):#dealing with user input
        doc = [doc]
    else:
        doc = doc.tolist()#dealing with dataset column (series)
    # Load pre trained ELMo model
    elmo = hub.Module("elmo_model", trainable=True)
    # create an instance of ELMo
    embeddings = elmo(inputs=doc,
        signature="default",
        as_dict=True)["elmo"]
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    texts_embedding=[]
    for i in range(len(doc)):
        words_embedding = sess.run(embeddings[i])
        texts_embedding.append(words_to_text_embedding(words_embedding))
        print('words_to_text_embedding',i)
    return texts_embedding


def save_extra_quran():
    data = pd.read_csv('quran.csv')
    data = data.iloc[:20]
    size = len(data['text'])
    stems=[]
    for aya_index in range(size):
        preprocessed_aya = preprocess_aya(data['text'][aya_index])
        stems.append(str(preprocessed_aya))
    data['stem']=stems
    data['embedding']=get_embedding_vector(data['text'])
    data.to_csv('quran_extra20.csv')



def save_dataset(data_source:pd.DataFrame, file_name="clean.csv"):
    # Saving the preprepared data in new csv file after putting output column at the end
    data_source.to_csv(file_name, encoding='utf8', index=False)

    # Saving the half
    corr = data_source.corr(method='kendall')
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(7, 5))
        ax = sns.heatmap(corr, mask=mask, vmax=.3,
                            square=True, annot=True, linewidths=.5)
    plt.savefig(file_name.replace('.csv','.png'))

def get_similarity(v1, v2):
    res = cosine_similarity(v1, v2)
    print(res)
    return res


def get_all_similarities(input_vector):
    data = pd.read_csv('quran_extra20.csv')
    size = len(data['text'])
    result=[]
    for index in range(size):
        print(data['embedding'][index], type(data['embedding'][index]))
        l = json.load(data['embedding'][index])
        print('---->',l, type(l))
        result.append((index, get_similarity(input_vector, data['embedding'][index]), data['text']))
    return result


def get_non_trainable_ELMO():
    tf=compat.v1
    tf.disable_eager_execution()
    #download the model to local so it can be used again and again
    elmo = hub.Module("model/", trainable=False)
    embeddings = elmo(
    ["the cat is on the mat", "what are you doing in evening"],
    signature="default",
    as_dict=True)["elmo"]
    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        message_embeddings = session.run(embeddings)
        print('**********')
        print(len(message_embeddings[0][0]))
        for i in message_embeddings:
            print(i,end='\n---\n')


def get_quran_vocabulary():
    data = pd.read_csv('cleaned.csv')
    #To store all Quran vocab into a single list 
    total_voc=[]
    for row in data['stem']:
        total_voc.extend( ast.literal_eval(row) )
    #Store each vocabulary with his correspondance occurence number
    dictionary={}
    for voc in total_voc:
        dictionary[voc]=total_voc.count(voc)
    #sorting dictionary in Desc order
    dictionary=sort_dict_by_value(dictionary,True)
    #saving dict to file
    to_txt('voacabulary.txt',dictionary)
    # with open('voacabulary.txt', 'w', encoding="utf-8") as f:
    #     for voc in dictionary:
    #         f.write(f"{voc}\n")


def prepare__train_data(dir_name):
    data = pd.read_csv('cleaned.csv')
    file_counter=0
    _data=[]
    for row in data['stem']:
        print(row)
        #Collecting flattened rows
        aya_tokens = " ".join(ast.literal_eval(row))
        _data.append(aya_tokens)        
        if len(_data)==6:
            to_txt(f'{dir_name}/{file_counter}.txt', data=_data)
            _data.clear()
            file_counter+=1


def to_txt(file_name, data):
    """
    Save any collection of data(list, dict, ...) into a txt file
    where each element in a single line
    """
    with open(file_name, 'a', encoding="utf-8") as f:
        for voc in data:
            f.write(f"{voc}\n")


def sort_dict_by_value(d, reverse = False):
  return dict(sorted(d.items(), key = lambda x: x[1], reverse = reverse))


def train_custom_elmo():
    data_train = pd.read_csv("swb/swb-train.csv")
    data_train['transcript'] = data_train['transcript'] + " ."
    print(data_train['transcript'].head())

    for i in range(0,data_train.shape[0],6):
        text = "\n".join(data_train['transcript'][i:i+6].tolist())
        fp = open("swb/train/"+str(i)+".txt","w")
        fp.write(text)
        fp.close()
    #Train
    data_dev = pd.read_csv("swb/swb-dev.csv")
    data_dev['transcript'] = data_dev['transcript'] + " ."
    
    for i in range(0,data_dev.shape[0],6):
        text = "\n".join(data_dev['transcript'][i:i+6].tolist())
        fp = open("swb/dev/"+str(i)+".txt","w")
        fp.write(text)
        fp.close()

    #Vocab
    texts = " ".join(data_train['transcript'].tolist())
    words = texts.split(" ")
    print("Number of tokens in Training data = ",len(words))
    dictionary = Counter(words)
    print("Size of Vocab",len(dictionary))
    sorted_vocab = ["<S>","</S>","<UNK>"]
    sorted_vocab.extend([pair[0] for pair in dictionary.most_common()])
    
    text = "\n".join(sorted_vocab)
    fp = open("swb/vocab.txt","w")
    fp.write(text)
    fp.close()



def cos_sim(sentence1_emb, sentence2_emb):
    """
    Cosine similarity between two columns of sentence embeddings
    
    Args:
      sentence1_emb: sentence1 embedding column
      sentence2_emb: sentence2 embedding column
    
    Returns:
      The row-wise cosine similarity between the two columns.
      For instance is sentence1_emb=[a,b,c] and sentence2_emb=[x,y,z]
      Then the result is [cosine_similarity(a,x), cosine_similarity(b,y), cosine_similarity(c,z)]
    """
    cos_sim = cosine_similarity(sentence1_emb, sentence2_emb)
    return np.diag(cos_sim)


# def get_s_transformer():
#     print('******************')
#     print(tf.config.list_physical_devices('GPU'))
#     print('--------')
#     # Load the English STSB dataset
#     stsb_dataset = load_dataset('stsb_multi_mt', 'en')
#     stsb_train = pd.DataFrame(stsb_dataset['train'])
#     stsb_test = pd.DataFrame(stsb_dataset['test'])
#     print('Check loaded data')
#     # Check loaded data
#     print(stsb_train.shape, stsb_test.shape)
#     stsb_test.head()

    # Load the pre-trained model
    # gpus = tf.config.list_physical_devices('GPU')
    # for gpu in gpus:
    #     # Control GPU memory usage
    #     tf.config.experimental.set_memory_growth(gpu, True)
    # module_url = 'https://tfhub.dev/google/universal-sentence-encoder/4'
    model = hub.load("USE/")

    # Generate Embeddings
    sentence1_emb = model(stsb_test['sentence1']).numpy()
    sentence2_emb = model(stsb_test['sentence2']).numpy()

    # Cosine Similarity
    stsb_test['USE_cosine_score'] = cos_sim(sentence1_emb, sentence2_emb)
    # sentences = ["This is an example sentence", "Each sentence is converted"]
    # model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1')
    # embeddings = model.encode(sentences)
    # print(embeddings)


def editIbnKathir():
    ibn_data = pd.read_csv('data/ibn_kathir.csv')
    quran_data  = pd.read_csv('data/quran.csv')

    #To clear missing rows
    ibn_data=ibn_data.iloc[:7683]
    ibn_data=ibn_data.loc[ibn_data['ID']!="`uid`"]
    data_size = ibn_data.shape[0]
    source_data=[]
    target_data=[]
    for index in range(data_size):
        current_row=ibn_data.iloc[index]
        try:
            source_data.append(quran_data.loc[quran_data['A'] == int(current_row['Source_Sura'])].loc[quran_data['B'] == int(current_row['Source_Verse']), 'text'].iloc[0])
            target_data.append(quran_data.loc[quran_data['A'] == int(current_row['target_sura'])].loc[quran_data['B'] == int(current_row['Target_verse']), 'text'].iloc[0])
        except Exception as e:
            print('Error', e)

    #append the new 2 columns to ibn_kathir data
    ibn_data['source']=source_data
    ibn_data['target']=target_data
    ibn_data.to_csv('ibn_kathir1.csv')


def get_fasttext_embedding(model:gensim.models.FastText, input:str)->list:
    input = preprocess_aya(input)
    text_embedding=[]
    for tag in input:
        text_embedding.append(model.wv.get_vector(tag))

    text_embedding = words_to_text_embedding(text_embedding)
    return text_embedding


def get_fastext_pearson(model:gensim.models.FastText,input1:str, input2:str)->float:
    input1 =  model. get_fasttext_embedding(model=model,input=input1)
    input2 = get_fasttext_embedding(model=model, input=input2)
    stat, p_value = pearsonr(input1, input2)
    return stat


def get_doc2vec_pearson(model:gensim.models.Doc2Vec,input1:str, input2:str)->float:
    print('###################')
    # model.similarity_unseen_docs
    input1 = preprocess_aya(input1)
    input2 = preprocess_aya(input2)
    print(input1)
    input1 = model.infer_vector(input1)
    input2 = model.infer_vector(input2)
    stat, p_value = pearsonr(input1, input2)
    return stat


def delete_additional_columns(data:pd.DataFrame, columns:list)->None:
    data.drop(columns, axis=1,inplace=True)

def fasttext_pearson_maker():
    data = pd.read_csv('data/cleaned.csv')
    ibn_data = pd.read_csv('ibn_kathir1.csv')
    prepared_data = [ast.literal_eval(d) for d in data['stem']]#data['text']#
    model = gensim.models.FastText(vector_size=200, window=25, alpha=0.025, workers=2)
    model.build_vocab(prepared_data)
    model.train(prepared_data, total_examples=data['stem'].shape[0], epochs=100)
    print(data.columns)
    data_size = ibn_data.shape[0]

    fasttext = []
    _pearson=0
    for index in range(data_size):
        _pearson = get_fastext_pearson(model=model, input1=ibn_data.iloc[index]['source'], input2=ibn_data.iloc[index]['target'])
        fasttext.append(_pearson)

    ibn_data['fasttext']=fasttext
    ibn_data.to_csv('ibn_kathir2.csv')


def tagged_document(list_of_list_of_words):
    for i, list_of_words in enumerate(list_of_list_of_words):
        yield gensim.models.doc2vec.TaggedDocument(ast.literal_eval(list_of_words), [i])


def doc2vec_pearson_maker():
    data = pd.read_csv('data/cleaned.csv')
    ibn_data = pd.read_csv('ibn_kathir2.csv')
    data_training = list(tagged_document(data['stem']))
    model = gensim.models.doc2vec.Doc2Vec(vector_size=130, min_count=2, epochs=30)
    model.build_vocab(data_training)
    data_size = ibn_data.shape[0]

    fasttext = []
    _pearson=0
    for index in range(data_size):
        _pearson = get_fastext_pearson(model=model, input1=ibn_data.iloc[index]['source'], input2=ibn_data.iloc[index]['target'])
        fasttext.append(_pearson)

    ibn_data['doc2vec']=fasttext
    ibn_data.to_csv('ibn_kathir2.csv')





# data = pd.read_csv('data/cleaned.csv')
# ibn_data = pd.read_csv('ibn_kathir2.csv')
# data_training = list(tagged_document(data['stem']))
# model = gensim.models.doc2vec.Doc2Vec(vector_size=130, min_count=2, epochs=30)
# model.build_vocab(data_training)
# data_size = ibn_data.shape[0]

# doc2vec = []
# _pearson=0
# for index in range(data_size):
#     print('1')
#     _pearson = get_doc2vec_pearson(model=model, input1=ibn_data.iloc[index]['source'], input2=ibn_data.iloc[index]['target'])
#     doc2vec.append(_pearson)

# ibn_data['doc2vec']=doc2vec
# ibn_data.to_csv('ibn_kathir2.csv')

#Code snippet to remove first aya of each sura
data = pd.read_csv('ibn_kathir2.csv')
print(data.shape)
data = data[data['Source_Verse']!=1]
data = data[data['Target_verse']!=1]
print(data.shape)
data.to_csv('test.csv')
columns_to_delete = set(data.columns)-{'fasttext','doc2vec', 'relevance_degree', 'common_roots'}
delete_additional_columns(data=data, columns=columns_to_delete)
save_dataset(data_source=data,file_name='ibn_kathir3.csv')
