from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import spacy
from nltk.corpus import stopwords
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("ru_core_news_sm")
# Загрузка и подгонка модели к отзывам
with open('reviews.txt', 'r', encoding='utf-8') as file:
    reviews = file.readlines()
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(reviews)


app = FastAPI()

stop_words = set(stopwords.words('russian'))



class TextRequest(BaseModel):
    text: str
#задание 1
@app.post('/task_1/')
async def process_text(request: TextRequest):
    text = request.text
    
    doc = nlp(text)
    tokens = [token.text for token in doc]
    print("TOKENIZE:",tokens)
    
    cleaned_tokens = [token.lower() for token in tokens if token.lower() not in stop_words]
    print("CLEANING:",cleaned_tokens)
    
    lemmatized_tokens = [token.lemma_ for token in nlp(" ".join(cleaned_tokens))]
    print("LEMMATIZE:",lemmatized_tokens)
    return {"tokens": lemmatized_tokens}


#функция для указания количества
async def search(query: str, top_n: int):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    return [reviews[i].strip() for i in top_indices]
#поиск отзывов
@app.post("/task_2/")
async def get_top_reviews(query: TextRequest):
    #3 - обратный массив
    results = await search(query.text,3)
    return {"results": results}


if __name__ == '__main__':
    import uvicorn
    # host="0.0.0.0"
    uvicorn.run(app, port=8000)

