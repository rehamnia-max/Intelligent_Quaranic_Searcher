# Code setup

1. create venv environment

```
virtualenv -p python3.10.4 venv
```

2. install project requirements

```
pip install -r requirements.txt
```

3. Run the Django project at any port

```
python manage.py runserver 8000
```

4- Use the API :
```
1- To Search in Quran basing on several machine learning algorithm:
http://127.0.0.1:8000/quran/

2- Prepare your input Aya by removing diacritics and stop words, segmentation, stemming....:
http://127.0.0.1:8000/quran/preprocess/Aya/

3- Get the vector representation using elmo-v3 model:
http://127.0.0.1:8000/quran/preprocess/Aya/

4- Get the cosine similarity between s1 and s2 (using elmo-v3 algorithm):
http://127.0.0.1:8000/quran/similarity/s1/s2/
```


5- Example of searching for "الموت" using fasttext which will return stemmer of similar words in Quran with the cosine similarity.

![example1](https://user-images.githubusercontent.com/66135457/209313226-f22136e8-b9c1-490e-bf07-dce804f55059.JPG)
![example2](https://user-images.githubusercontent.com/66135457/209313598-22ef8447-0365-4ac5-8a7b-24285dcefe91.JPG)


5- Feel free to contact me anytime
```
rehamniawalid.info@gmail.com
```
</br>
</br>
