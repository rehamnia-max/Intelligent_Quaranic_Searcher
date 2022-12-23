from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('preprocess/<str:text>/', views.preprocess),
    path('embedding/<str:text>/', views.embedding),
    path('similarity/<str:s1>/<str:s2>/', views.get_similarity),
    path('testTemplate/', views.index),
    path('doc2vec/', views.doc_to_vec),
    path('fasttext', views.get_fasttext),
]