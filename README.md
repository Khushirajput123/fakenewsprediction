# Fake News Detection using Machine Learning
This project builds a **machine learning model to classify news articles as REAL or FAKE** using Natural Language Processing (NLP). The model processes news text, converts it into numerical features using **TF-IDF**, and trains a **Logistic Regression classifier** to make predictions.
## Dataset
The dataset contains news articles with the following columns:
* **title** – headline of the news article
* **text** – full article content
* **label** – indicates whether the news is **REAL or FAKE**
For better analysis, the **title and text are combined** into a single feature called **content**.
---
## Libraries Used
### Pandas
Used for **loading and handling the dataset** in tabular form.
### NumPy
Used for **numerical operations and array handling**.
### re (Regular Expressions)
Used to **clean text data** by removing numbers, punctuation, and special characters.
### NLTK
A Natural Language Processing library used for **text preprocessing**.
Important components used:
* **Stopwords** – removes common words such as *the, is, and* that do not add meaning.
* **PorterStemmer** – reduces words to their root form (example: *playing → play*).
### TfidfVectorizer
Converts text data into **numerical vectors** so that machine learning models can understand it.
TF-IDF measures how important a word is in a document compared to the entire dataset.
### train_test_split
Splits the dataset into:
* **Training data** (used to train the model)
* * **Testing data** (used to evaluate model performance)
### accuracy_score
Calculates how many predictions were correct.
---
## Machine Learning Model: Logistic Regression
This project uses **Logistic Regression** for classification.
Logistic Regression is commonly used for **binary classification problems**, where the output has two classes.
In this project:
* **0 → Fake News**
* **1 → Real News**
The model learns patterns in the text and calculates the **probability that a news article is real or fake**.
### Why Logistic Regression?
* Works well for **text classification problems**
* Efficient for **large datasets**
* Easy to interpret and fast to train

## Project Workflow
1. Load the dataset
2. Combine news **title and text** into one feature
3. Clean the text (remove special characters, stopwords, apply stemming)
4. Convert text into numerical form using **TF-IDF**
5. Split data into **training and testing sets**
6. Train the **Logistic Regression model**
7. Evaluate model performance using **accuracy score**
## Model Evaluation
The model performance is measured using **training accuracy and testing accuracy**, which show how well the model predicts fake and real news articles.
Objective
The goal of this project is to demonstrate a **complete NLP machine learning pipeline**, including text preprocessing, feature extraction, model training, and evaluation.



