"""
Sentiment Analysis Model
Trains on labeled text data using TF-IDF + Logistic Regression / Naive Bayes
"""
import re
import string
import pickle
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,classification_report,confusion_matrix
from sklearn.pipeline import Pipeline

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    nltk.download("stopwords",quiet=True)
    STOP_WORDS=set(stopwords.words("english"))
    STEMMER=PorterStemmer()
    NLTK_AVAILABLE=True
except ImportError:
    STOP_WORDS={"i","me","my","myself","we","our","ours","ourselves","you","your","yours","yourself","he","him","his","himself","she","her","hers","herself","it","its","itself","they","them","their","theirs","themselves","what","which","who","whom","this","that","these","those","am","is","are","was","were","be","been","being","have","has","had","having","do","does","did","doing","a","an","the","and","but","if","or","because","as","until","while","of","at","by","for","with","about","against","between","into","through","during","before","after","above","below","to","from","up","down","in","out","on","off","over","under","again","further","then","once","here","there","when","where","why","how","all","both","each","few","more","most","other","some","such","no","nor","not","only","own","same","so","than","too","very","s","t","can","will","just","don","should","now","d","ll","m","o","re","ve","y","ain","aren","couldn","didn","doesn","hadn","hasn","haven","isn","ma","mightn","mustn","needn","shan","shouldn","wasn","weren","won","wouldn"}
    STEMMER=None
    NLTK_AVAILABLE=False

def preprocess(text:str,stem:bool=False)->str:
    text=text.lower()
    text=re.sub(r"http\S+|www\S+|https\S+","",text)
    text=re.sub(r"@\w+","",text)
    text=re.sub(r"#","",text)
    text=re.sub(r"<.*?>","",text)
    text=re.sub(r"\d+","",text)
    text=text.translate(str.maketrans("","",string.punctuation))
    tokens=text.split()
    tokens=[t for t in tokens if t not in STOP_WORDS and len(t)>1]
    if stem and STEMMER:
        tokens=[STEMMER.stem(t) for t in tokens]
    return " ".join(tokens)

def preprocess_corpus(texts,stem:bool=False):
    return [preprocess(t,stem=stem) for t in texts]

SAMPLE_DATA={
"texts":[
"I absolutely love this product! It works perfectly.",
"Amazing experience, will definitely buy again.",
"This is the best purchase I've ever made, totally worth it!",
"Exceeded all my expectations. Five stars!",
"Fantastic quality and super fast delivery. Very happy!",
"Great value for money. Highly recommend to everyone.",
"I'm so happy with this. Works exactly as described.",
"Outstanding customer service and a brilliant product.",
"Loved every bit of it. Makes life so much easier!",
"Brilliant! Does exactly what it says on the tin.",
"Really impressed by the quality. Will order again.",
"Superb product. Arrived on time and in perfect condition.",
"This made my day! So glad I bought it.",
"Top notch quality, very comfortable and stylish.",
"Wonderful product, couldn't be happier with my purchase.",
"The best I've used in years. Highly satisfied.",
"Wow, just wow. Absolutely stunning results.",
"Very pleased with my order. Prompt delivery too.",
"Excellent! Just what I needed. Perfect fit.",
"So good! Lightweight, durable, and looks great.",
"Terrible product. Broke after two days of use.",
"Worst purchase I've ever made. Complete waste of money.",
"Arrived damaged and customer support was useless.",
"Absolutely disappointed. Nothing like the description.",
"Poor quality. Fell apart immediately. Do not buy.",
"Very unhappy with this. Returning it immediately.",
"Doesn't work at all. Cheap and poorly made.",
"Awful experience from start to finish. Avoid!",
"Not worth a single cent. Total garbage.",
"Horrible product, horrible service. Stay away.",
"Complete junk. Broke within a week.",
"I regret buying this. It's useless.",
"Defective on arrival. No refund offered either.",
"Misleading photos. Product looks nothing like advertised.",
"Extremely disappointed. Cheap materials, bad finish.",
"Waste of time and money. Would give zero stars.",
"The product stopped working after first use. Terrible.",
"Absolutely appalling quality. Not fit for purpose.",
"Never buying from this seller again. Dreadful.",
"Cheap knock-off. Looked nothing like the pictures.",
"The product is okay. Nothing special but does the job.",
"Average quality. Arrived on time at least.",
"It's fine, nothing to rave about but not bad either.",
"Decent product for the price. Could be improved.",
"Mediocre. Works sometimes, not always reliable.",
"It's an okay product. I've seen better and worse.",
"Not the best, not the worst. Gets the job done.",
"Functional but unimpressive. Packaging was good though.",
"Mixed feelings. Some features are good, others are lacking.",
"Acceptable. Meets basic requirements but nothing more."
],
"labels":["positive"]*20+["negative"]*20+["neutral"]*10
}

class SentimentAnalyser:
    CLASSIFIERS={
        "naive_bayes":MultinomialNB(),
        "logistic_regression":LogisticRegression(max_iter=1000,random_state=42)
    }
    VECTORIZERS={
        "tfidf":TfidfVectorizer(ngram_range=(1,2),max_features=10000),
        "count":CountVectorizer(ngram_range=(1,2),max_features=10000)
    }
    def __init__(self,classifier:str="logistic_regression",vectorizer:str="tfidf",stem:bool=False):
        if classifier not in self.CLASSIFIERS:
            raise ValueError(f"classifier must be one of {list(self.CLASSIFIERS)}")
        if vectorizer not in self.VECTORIZERS:
            raise ValueError(f"vectorizer must be one of {list(self.VECTORIZERS)}")
        self.classifier_name=classifier
        self.vectorizer_name=vectorizer
        self.stem=stem
        self.pipeline=Pipeline([("vec",self.VECTORIZERS[vectorizer]),("clf",self.CLASSIFIERS[classifier])])
        self.classes_=None
        self.metrics={}

    def train(self,texts,labels,test_size:float=0.2):
        print("\n"+"="*55)
        print(" Sentiment Analysis — Training Pipeline")
        print("="*55)
        clean=preprocess_corpus(texts,stem=self.stem)
        dist=Counter(labels)
        print("Label distribution:",dict(dist))
        X_tr,X_te,y_tr,y_te=train_test_split(clean,labels,test_size=test_size,random_state=42,stratify=labels)
        self.pipeline.fit(X_tr,y_tr)
        self.classes_=list(self.pipeline.classes_)
        y_pred=self.pipeline.predict(X_te)
        acc=accuracy_score(y_te,y_pred)
        f1=f1_score(y_te,y_pred,average="weighted")
        self.metrics={
            "accuracy":round(acc,4),
            "f1_weighted":round(f1,4),
            "report":classification_report(y_te,y_pred),
            "confusion_matrix":confusion_matrix(y_te,y_pred).tolist(),
            "classes":self.classes_
        }
        print("Accuracy:",acc)
        print(classification_report(y_te,y_pred))
        return self.metrics

    def predict(self,text:str)->dict:
        clean=preprocess(text,stem=self.stem)
        label=self.pipeline.predict([clean])[0]
        try:
            probs=self.pipeline.predict_proba([clean])[0]
            prob_map={c:round(float(p),4) for c,p in zip(self.classes_,probs)}
        except AttributeError:
            prob_map={c:(1.0 if c==label else 0.0) for c in self.classes_}
        confidence=prob_map.get(label,1.0)
        return {"label":label,"confidence":confidence,"probabilities":prob_map,"cleaned_text":clean}

    # Added Feature
    def predict_batch(self,text_list):
        return [self.predict(t) for t in text_list]

    def save(self,path:str="sentiment_model.pkl"):
        with open(path,"wb") as f:
            pickle.dump(self,f)
        print("Model saved →",path)

    @staticmethod
    def load(path:str="sentiment_model.pkl"):
        with open(path,"rb") as f:
            model=pickle.load(f)
        print("Model loaded ←",path)
        return model

if __name__ == "__main__":
    print("\nRunning quick test using SAMPLE_DATA...\n")

    texts = SAMPLE_DATA["texts"]
    labels = SAMPLE_DATA["labels"]

    model = SentimentAnalyser()
    model.train(texts, labels)

    test_text = "This product is amazing and works perfectly!"
    result = model.predict(test_text)

    print("\nTest Text:", test_text)
    print("Prediction:", result)