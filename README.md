# AI and Human Text Detection using NLP

## Project Overview
The AI and Human Text Detection project uses Natural Language Processing (NLP) and Machine Learning to classify text as AI-generated or human-written. The main objective is to analyze text data, process it, and train a machine learning model to predict the author category.

## Dataset
The project uses an AI and Human Text Dataset downloaded through KaggleHub.

- **Text Column:** Text
- **Target Column:** Author
- **Processed Text Column:** Processed_Text

The dataset is analyzed to identify missing values, duplicate records, and the distribution of different author categories.

## Technologies Used
- Python
- Pandas
- NumPy
- NLTK
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- KaggleHub

## Project Workflow

1. **Data Loading:** Download and load the dataset into a Pandas DataFrame.
2. **Data Cleaning:** Check missing values and duplicate records.
3. **Data Visualization:** Use count plots and pie charts to understand the distribution of author labels.
4. **Text Preprocessing:** Convert text to lowercase, remove URLs and HTML tags, remove unwanted characters, and normalize whitespace.
5. **Tokenization:** Split the text into individual words.
6. **Stopword Removal:** Remove common English words that may contribute less to classification.
7. **Lemmatization:** Convert words into their base forms using NLTK.
8. **Label Encoding:** Convert author labels into numerical values using LabelEncoder.
9. **TF-IDF Vectorization:** Convert processed text into numerical features using TfidfVectorizer.
10. **Model Training:** Train a Logistic Regression classifier.
11. **Model Evaluation:** Evaluate the model using accuracy, precision, recall, and F1-score.
12. **Model Saving:** Save the trained model, label encoder, and TF-IDF vectorizer using Joblib.

## Machine Learning Model

The project uses **Logistic Regression** for classification. The model learns patterns from the TF-IDF features and predicts the author category of the input text.

The model is evaluated using:
- Accuracy Score
- Classification Report
- Precision
- Recall
- F1-Score

## Model Deployment Files

The following files can be saved for future predictions:

- `logistic_regression_model.joblib` — Stores the trained classification model.
- `label_encoder.joblib` — Stores the label encoding information.
- `tfidf_vectorizer.joblib` — Stores the fitted TF-IDF vectorizer.

These files allow the trained model to be loaded without training it again.

## Installation

Install the required Python libraries using the following command:

```bash
pip install pandas numpy nltk scikit-learn matplotlib seaborn joblib kagglehub
```

## How to Run

1. Download or clone the project repository.
2. Open `AI_and_Human_Text_Dataset.ipynb` in Jupyter Notebook or VS Code.
3. Install the required libraries.
4. Run the notebook cells in the correct order.
5. Download and preprocess the dataset.
6. Train and evaluate the Logistic Regression model.
7. Save the model files for future use.

## Applications

- Educational NLP projects
- Text classification
- AI-generated and human-written text analysis
- Machine learning experimentation
- Development of text detection applications

## Limitations

AI and human text detection is not always completely accurate. The model may incorrectly classify some texts because AI-generated and human-written content can have similar writing patterns. The prediction should therefore be treated as an estimate rather than definitive proof of authorship.

## Conclusion

This project demonstrates how NLP techniques and machine learning can be combined to classify text. Text preprocessing, TF-IDF vectorization, Label Encoding, and Logistic Regression form the main workflow. The trained model can be saved and reused for future text predictions.

## License

Check the original dataset's license and usage conditions on Kaggle before redistributing the dataset or using it for other purposes.
