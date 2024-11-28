import os
# os.system("python -m venv venv")
# os.system("venv/Scripts/activate")
os.system("python -m nltk.downloader stopwords")
os.system("pip install -r requirements.txt")
os.system("python -m spacy download ru_core_news_sm")