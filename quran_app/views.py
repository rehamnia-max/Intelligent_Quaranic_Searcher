from django.http import HttpResponse
from utils import *
import json
from django.shortcuts import render
import pandas as pd
import gensim
import ast 

def preprocess(request, text):
    preprocessed_text = preprocess_aya(text)
    json_stuff = json.dumps({"Aya stemmed" : preprocessed_text},ensure_ascii=False).encode('utf8')   
    return HttpResponse(json_stuff)


def embedding(request, text):
    preprocessed_text = get_embedding_vector(text)
    return HttpResponse(preprocessed_text)

def get_similarity(request, s1,s2):
    embedding1 = get_embedding_vector(s1)
    embedding2 = get_embedding_vector(s2)
    similarity = cos_sim(embedding1,embedding2)
    return HttpResponse(similarity)

def index(request):
    input =  request.GET.get('search',default='')
    input_type =  request.GET.get('input_type',default='text')
    result={}
    preprocessed_text=[]
    if input!='':
        if input_type=='text':
            preprocessed_text = preprocess_aya(input)
            result = get_text_similarities(preprocessed_text)
        elif input_type=='word':
            preprocessed_text = preprocess_aya(input)
            result = get_word_similarities(preprocessed_text)
        elif input_type=='fasttext':
            preprocessed_text = preprocess_aya(input)
            result = get_fasttext(input=input)


    return render(request= request,template_name='index.html', 
        context={'text':input, 'preprocessed':preprocessed_text, 'result':result})


def doc_to_vec(request):
    return
    return HttpResponse(model.wv.similar_by_word('موسى'))
    # return HttpResponse(model.wv.vocab)


def tagged_document(list_of_list_of_words):
    for i, list_of_words in enumerate(list_of_list_of_words):
        yield gensim.models.doc2vec.TaggedDocument(ast.literal_eval(list_of_words), [i])


def get_text_similarities(input:list)->dict:
    """
    Doc2Vec
    """
    #Training the model 
    data = pd.read_csv('data/cleaned.csv')
    data_training = list(tagged_document(data['stem']))
    model = gensim.models.doc2vec.Doc2Vec(vector_size=130, min_count=2, epochs=30)
    model.build_vocab(data_training)
  
    sim = {}
    for aya in data['stem']:
        sim[aya]=model.similarity_unseen_docs(input, ast.literal_eval(aya))
    sim=sort_dict_by_value(sim,True)
    return sim


def get_word_similarities(input:list)->dict:
    """
    Doc2Vec
    """
    #Training the model 
    data = pd.read_csv('data/cleaned.csv')
    data_training = list(tagged_document(data['stem']))
    model = gensim.models.doc2vec.Doc2Vec(vector_size=40, min_count=2, epochs=30)
    model.build_vocab(data_training)
    sim = model.wv.most_similar(positive=input)
    return sim


def get_fasttext(input:list):
    print('######################')
    data = pd.read_csv('data/cleaned.csv')
    prepared_data = [ast.literal_eval(d) for d in data['stem']]#data['text']#
    model = gensim.models.FastText(vector_size=200, window=25, alpha=0.025, workers=2)
    model.build_vocab(prepared_data)
    model.train(prepared_data, total_examples=data['stem'].shape[0], epochs=100)
    print(data.columns)
    rs = model.wv.get_vector('الحمد')
    print(rs)
    return model.wv.most_similar(positive=input)
