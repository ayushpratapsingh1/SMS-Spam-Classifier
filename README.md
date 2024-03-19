# SMS-Spam-Classifier
SMS Spam Classifier: Identifies spam vs. ham messages using machine learning techniques.
<br>
In the SMS Spam Classifier project, various machine learning algorithms were employed, including Bernoulli Naive Bayes, Multinomial Naive Bayes, Gaussian Naive Bayes, Linear Regression, Adaboost, and Random Forest. Among these algorithms, Multinomial Naive Bayes demonstrated exceptional performance with an accuracy of 97%, surpassing the other methods.
Steps in the SMS Spam Classifier Project:
- Data Preprocessing: Cleaning and preparing the SMS dataset for analysis.
- Feature Extraction: Utilizing techniques like TF-IDF Vectorizer and CountVectorizer to convert text data into numerical features.
- Model Training: Training the classifiers (Bernoulli NB, Multinomial NB, etc.) on the preprocessed data.
- Model Evaluation: Assessing the performance of each algorithm using metrics like accuracy.
- Model Selection: Identifying Multinomial Naive Bayes as the best-performing algorithm.
Key Components and Parameters:
- TF-IDF Vectorizer: Transforms text data into numerical vectors based on term frequency-inverse document frequency.
- CountVectorizer: Converts a collection of text documents into a matrix of token counts.
- Pickle: Used for serializing and deserializing Python objects to save trained models.


Installations in vscode terminal: scikit-learn,nltk,streamlit,
For webview we used streamlit.
