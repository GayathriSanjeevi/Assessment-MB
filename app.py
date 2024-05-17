from flask import Flask, render_template, request #importing flask for building API

from urllib.error import HTTPError, URLError#
import urllib.request #importing urllib.request for retrieving data using url from web
from bs4 import BeautifulSoup #to extract data from html
import nltk #for processing text from html

from sklearn.feature_extraction.text import TfidfVectorizer #to convert text to numbers that can be used for learning
from sklearn.metrics.pairwise import cosine_similarity#to find the similarity between the query input by the user and the documents in the corpus

app = Flask(__name__)#creating an instance of flask

@app.route('/')

def index():#chat.html - web page that interacts with the user - UI
   return render_template('chat.html',result=None)

@app.route('/get',methods=["POST","GET"])#sending message to chatbot
def get():
   if request.method=="POST":
      msg = request.form.get('Questions')#getting the input from user interface
      input=msg
      response_final=get_chat_response(input) #function for replying to the question asked
      return render_template('chat.html',result=response_final)#printing the result in user interface
#Data Fetch and Tokenisation using NLTK to use in chatbot
#Retrieving the data from a wikipedia page -- separating it into sentence tokens for comapring it with the input query
def train_data():
   # URL of the web page to fetch 
   url ='https://en.wikipedia.org/wiki/Python_(programming_language)'
   try:
    # Open the URL and read its content 
      file = urllib.request.urlopen(url) 
    # Read the content of the response 
      html = file.read() 
      html_string = html.decode("utf-8")
   except HTTPError as e:
      print("HTTP error")
   except URLError as e:
      print("Server error")
   #using beautiful soup for parsing html document --- to navigate and search data
   soup=BeautifulSoup(html_string,'html.parser')

   text=' ' 
   for paragraph in soup.find_all('p'):
      text+=paragraph.text #extracting paragrah wise using soup.find_all
   text=text.lower()#converting the text to lower case
   sentence_tokens=nltk.sent_tokenize(text) #paragragh are tokenised into sentences
   return sentence_tokens


def get_chat_response(user_input):
   bot_response=" "
   tokens=train_data()#creating sentence tokens
#Preprocess and vectorize data available
   vectorizer = TfidfVectorizer()
   text_vectorized = vectorizer.fit_transform(tokens)

# Preprocess and vectorize user input
   user_input_vectorized = vectorizer.transform([user_input])

# Compute cosine similarity between user input and predefined responses
   similarity_scores = cosine_similarity(user_input_vectorized, text_vectorized)

# Get index of the most similar response
   most_similar_index = similarity_scores.argmax()

# Retrieve the most similar response
   bot_response = tokens[most_similar_index]
   return bot_response#return the response

if __name__ == '__main__':
   app.run(debug=True)