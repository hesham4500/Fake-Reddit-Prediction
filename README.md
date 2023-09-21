

---

## **Problem Defination:**
The spread of misinformation on social media platforms is an ever-growing problem. Organizations, politicians, individuals looking for personal gain and even certain news media outlets engage in propagating fake news to sway people's decisions as well as distorting events to fit a bias or prejudice. The degree of authenticity of the news posted online cannot be definitively measured, since the manual classification of news is tedious and time-consuming and is also subject to bias. To tackle the growing problem, detection, classification and mitigation tools are a need of the hour.

---

## **Methodology**
The categories, bs (i.e. bullshit), junksci(i.e. junk science), hate, fake, conspiracy, bias, satire and state declare the
category under which untrustworthy or false news fall under. 

#### **1. Load Dataset**

#### **2. Text Preprocessing**
* Taking care of null/missing values .
* Text Translation to English.
* Remove links,tags, single letters, and numbers.
* Uppercase to lowercase.
* Tokenization.
* Stop Word Removal, Stemming using the Natural Language Toolkit Library. 

### **3. Visualization**
* Draw WordCloud.
* Bar plot for number of words in each text.
* N-Gram Analysis and plotting using bars.

#### **4. Feature engineering**
1. Using TF-IDF
2. Using CountVectorizer

#### **5. TF-IDF Modeling**

a. Using models without kfolds: 

* BernoulliNB
* MultinomialNB
* GaussianNB
* CalibratedClassifierCV
* LogisticRegression
* RandomForestClassifier
* XGBClassifier
* MLPClassifier

b. Using RandomSearch with Pipline:

 * LogisticRegression 
 * RandomForestClassifier

c. Using BaysianSearch With pipline:

 * XGBClassifier
 * MultinomialNB

#### **6. CountVectorizer Modeling**
a. Using models without kfolds: 

* BernoulliNB
* MultinomialNB
* CalibratedClassifierCV
* LogisticRegression
* RandomForestClassifier
* XGBClassifier
* MLPClassifier

b. Using RandomSearch with Pipline:

 * LogisticRegression 
 * RandomForestClassifier

c. Using BaysianSearch With pipline:

 * XGBClassifier
 * MultinomialNB

#### **7.Train best Model on translated text**

#### **8. Predict the test set and submit**
---
## **Input:**

The input for the model is a dataset for the competition consists of two CSV files: `xy_train.csv` with 60,000 rows representing the training data with labels, and `x_test.csv` with 59,151 rows representing the test data without labels.

---

## **Output:**
The output of the fake news detection problem on Reddit is a binary classification model that can accurately predict whether a given Reddit post is real or fake news based on its title and other relevant information.

---
## **Required Data Mining Function:**

The required data mining function for the fake news detection problem on Reddit involves a combination of feature engineering, model selection and training, hyperparameter tuning, and evaluation. 

### **Feature engineering:**
- TF-IDF and CountVectorizer.

#### **Model selection and training:**
* BernoulliNB
* MultinomialNB
* GaussianNB
* CalibratedClassifierCV
* LogisticRegression
* RandomForestClassifier
* XGBClassifier
* MLPClassifier

#### **Hyperparameters Tunning:**
* Using RandomSearch
* Using BayesSearch

#### **Evaluations:**

* Calculating accuracy
* Calculating F1 score
* Calculating AUC
* Calculating Confusion Matrix

---

## **Challenges**


The fake news detection problem on Reddit poses several challenges, including:

- **Ensuring data quality**, dealing with missing data, outliers, and ensuring representative data.

- **Overfitting**, preventing the model from fitting noise in the training data.

- **Model complexity**, selecting the appropriate architecture and hyperparameters.

- **Interpretability**, understanding the model's predictions and important features.

- **Text preprocessing:** Text data can be messy and require extensive preprocessing, including handling null/missing values, translating text to English, removing links and tags, and tokenization.

- **Model selection and hyperparameter tuning:** Choosing an appropriate machine learning algorithm for binary classification, and tuning its hyperparameters, can be a challenging task.

---
## **Impact**

The impact of successfully detecting fake news on Reddit can be significant.

- It can help prevent the spread of misinformation and promote transparency and reliability in the information shared on social media platforms. By accurately detecting and flagging fake news posts, users can be more informed and better equipped to make decisions based on trustworthy information.

- It can help protect individuals and communities from the negative consequences of fake news, such as political polarization, social unrest, and economic harm. By identifying and countering false narratives, the impact of fake news can be minimized.

- It can help maintain the integrity of the Reddit platform and other social media platforms. If users perceive that the platform is rife with fake news and misinformation, they may be less likely to use it or trust the information shared on it.

Overall, the impact of successfully detecting fake news on Reddit can be far-reaching and have positive implications for individuals, communities, and society as a whole.


---
## **Ideal Solution:**

- The ideal solution for the fake news detection problem on Reddit involves creating a Naive Bayes model with a Laplace smoothing parameter of 1.0 and a moderate selection of informative features , trained on a labeled dataset using TF-IDF as the feature extraction technique.

- The resulting model achieves a score of **auc = 0.8517**. This model can be used to predict whether a given Reddit post is real or fake news based on its features, with a high degree of accuracy.


