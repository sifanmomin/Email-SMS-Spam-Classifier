##Email/SMS Spam Classifier
## app demo - https://email-sms-spam-classifier-qwe6xxspikhlqf3piyejmb.streamlit.app/#spam
This repository contains a machine learning project aimed at classifying emails and SMS messages as spam or not spam (ham). The project leverages natural language processing (NLP) techniques and machine learning algorithms to build an effective spam detection system.

#Project Overview

The primary goal of this project is to develop a robust spam classifier using various machine learning approaches. The key highlights of the project include:

Naive Bayes Classifier: Initial experiments using the Multinomial Naive Bayes (MNB) classifier.

TF-IDF Vectorization: Improved text feature extraction with Term Frequency-Inverse Document Frequency (TF-IDF).

Model Optimization: Explored ensemble methods such as Voting Classifier and Stacking to enhance performance.

Final Model: Selected the MNB classifier with TF-IDF due to its optimal performance.

#Dataset

The dataset used for this project was sourced from the UCI ML repository, containing labeled SMS and email messages as spam or ham.

#Data Preprocessing

Removed missing values and duplicates.

Performed text preprocessing, including:

Lowercasing

Removing special characters, punctuation, and stopwords

Tokenization

Lemmatization

Methodology

#1. Exploratory Data Analysis (EDA)

Analyzed the distribution of spam vs. ham messages.

Visualized common words in both spam and ham messages.

#2. Feature Extraction

Utilized TF-IDF vectorizer to transform text data into numerical format.

#3. Model Building

Experimented with multiple classifiers:

Multinomial Naive Bayes (MNB)

Voting Classifier

Stacking Classifier

Evaluated model performance using metrics like accuracy, precision, recall, and F1-score.

#4. Final Model

The MNB classifier with TF-IDF vectorization was chosen as the final model due to its simplicity and efficiency.

Results

The final model achieved:

High accuracy and F1-score.

Good balance between precision and recall.

Deployment

The project was deployed using Streamlit, enabling an interactive web interface for users to input messages and classify them as spam or ham.

#How to Run

Clone the repository:

git clone [https://github.com/your-username/email-sms-spam-classifier.git](https://github.com/sifanmomin/Email-SMS-Spam-Classifier.git)

Install dependencies:

pip install -r requirements.txt

Run the Streamlit app:

streamlit run app.py

Open the provided URL in your browser to access the app.

Folder Structure

email-sms-spam-classifier/
├── data/
│   └── spam_dataset.csv       # Dataset
├── notebooks/
│   └── eda.ipynb              # Jupyter Notebook for EDA
├── models/
│   └── final_model.pkl        # Saved final model
├── app.py                     # Streamlit app
├── requirements.txt           # Dependencies
├── README.md                  # Project description

Technologies Used

Python

Scikit-learn

NLTK

Streamlit

Pandas

Matplotlib

Seaborn

Future Enhancements

Incorporate deep learning models (e.g., LSTM, BERT) for better performance.

Expand the dataset with more diverse samples.

Add language support for non-English messages.
