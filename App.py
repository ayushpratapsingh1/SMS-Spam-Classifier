import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

ps=PorterStemmer()

def transform_text(text):
    text = text.lower()
    text=nltk.word_tokenize(text)#breaking down the text to small list of of words
    
    y=[]
    for i in text:
        if i.isalnum():#remove special chars
            y.append(i)
    
    text=y[:]#assign y to text
    y.clear()#clear y
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:# removing stopwords and punctuation
            y.append(i)
    
    text=y[:]#assign y to text
    y.clear()#clear y
    
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)# Joining of list of data into a single string

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))


# Define custom CSS styles
custom_styles = """
    <style>
        .title-wrapper {
            text-align: left;
            padding-top: 10px;
            
        }
        .logo {
            position: absolute;
            color: #E8E8E8;
            font-size: 24px;
            font-weight: bold;
            border: 2px solid #D20103;
            padding: 5px 10px;
            border-radius: 10px;

        }
    </style>
"""
# Inject custom CSS styles
st.markdown(custom_styles, unsafe_allow_html=True)

# Add logo or text
st.write('<div class="logo"><b>APS</b></div>', unsafe_allow_html=True)
st.title("SMS spam Classifier")
input_sms=st.text_area("Enter the message")

if st.button('Classify'):
    if input_sms.strip():  # Check if input is not empty
        # Preprocess
        transformed_sms = transform_text(input_sms)
        # Vectorize
        vector_input = tfidf.transform([transformed_sms])
        # Predict
        result = model.predict(vector_input)[0]
        # Display result
        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")
    else:
        st.warning("Please enter a message to classify.")