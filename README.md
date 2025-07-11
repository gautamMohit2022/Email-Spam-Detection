#  Email Spam Detection using Machine Learning

This project focuses on identifying **spam vs. non-spam (ham)** emails using text classification techniques. It uses a labeled dataset of emails and applies machine learning algorithms to build a spam detector.

---

##  Objective

Build a classification model to automatically detect spam emails using Natural Language Processing (NLP) and machine learning.

---


---

##  Technologies Used

- Python
- scikit-learn
- Pandas, NumPy
- NLP (CountVectorizer, TF-IDF)
- Naive Bayes Classifier
- Jupyter Notebook

---

##  Process Workflow

1. **Data Loading**: Read and explore the `spam.csv` dataset.
2. **Text Preprocessing**:
   - Lowercasing
   - Removing punctuation, stopwords
   - Tokenization
3. **Feature Extraction**:
   - Bag of Words
   - TF-IDF Vectorization
4. **Model Training**:
   - Trained Naive Bayes classifier
   - Evaluated using accuracy, precision, recall
5. **Prediction**:
   - Model predicts whether a message is spam or not

---

##  Dataset Info

- **Columns**:
  - `label` → spam or ham
  - `message` → actual email content
- ~5,500 labeled messages

---

##  How to Run

```bash
# Install dependencies (if needed)
pip install pandas scikit-learn numpy

# Run the notebook
jupyter notebook Email_Spam_Detection_with_Machine_Learning.ipynb


Future Improvements
Use advanced models like SVM, Logistic Regression, or LSTM

Add deployment via Streamlit or Flask

Include real-time email text input for prediction


Author:
Mohit Gautam
GitHub: @gautamMohit2022


