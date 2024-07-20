* Overview: This repository contains a machine learning model designed to classify SMS and email messages as either spam or not spam (ham). 
            The model leverages natural language processing (NLP) techniques and supervised learning algorithms to accurately detect unwanted messages, providing a 
            reliable tool for filtering and managing digital communications.

* Steps for Model Development:
    - Data Preprocessing: Cleans and preprocesses text data, including tokenization, stopword removal, and stemming.
    - Feature Extraction: Utilizes techniques such as TF-IDF (Term Frequency-Inverse Document Frequency) to convert text into numerical features suitable for machine 
                          learning.
    - Model Training: # Implements and trains various machine learning models by giving data using different methods  
                      # Naive Bayes : GaussianNB, MultinomialNB, BernoulliNB, Support Vector Machines (SVM), Logistic Regression, Decision Tree Classifier, 
                        KNeighborsClassifier, Random Forest Classifier, AdaBoost Classifier, Bagging Classifier, Extra Trees Classifier, Gradient Boosting Classifier, 
                        XGBoost Classifier.
                      # Also Implemented Soft Voting Classifier and Stacking Classifier with estimators Multinomial NB, Random Forest, Extra Trees Classifier 
                      # Also tried out by applying Scaling (MinmaxScaling)
                      # Different datasets were tried out, including using all features and feeding the model with the most 3000 common words..
    - Evaluation: Assesses model performance using metrics such as accuracy, precision and confusion Matrix to ensure robustness and reliability.
    - Results: The final model, Multinomial Naive Bayes, with  achieves an accuracy of 97.09% and a precision score of 1 on the test datasets, demonstrating its 
               effectiveness in identifying spam messages.
               The dataset with the 3000 most common words was chosen as features to feed the model.
              (Also achieving the same results with Voting Classifier)  

* Libraries Used:
               - pandas: Data manipulation and analysis.
               - nltk: Natural Language Toolkit for text processing.
               - numpy: Numerical computing.
               - sklearn: Machine learning algorithms and tools.
               - seaborn: Data visualization.
               - streamlit: Deployment and interactive web applications.

* Tools Used:
             - Google Colab: Used for model development and training.
             - PyCharm: Used for giving a site view and additional development.  

         
