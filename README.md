Overview: This repository contains a machine learning model designed to classify SMS and email messages as either spam or not spam (ham). 
The model leverages natural language processing (NLP) techniques and supervised learning algorithms to accurately detect unwanted messages, providing a reliable tool for filtering and managing digital communications.

Features: Data Preprocessing: Cleans and preprocesses text data, including tokenization, stopword removal, and stemming.

Feature Extraction: Utilizes techniques such as TF-IDF (Term Frequency-Inverse Document Frequency) to convert text into numerical features suitable for machine 
                    learning.

Model Training: Implements and trains various machine learning models by giving data using different methods  
                Naive Bayes : GaussianNB, MultinomialNB, BernoulliNB, Support Vector Machines (SVM), Logistic Regression, Decision Tree Classifier, 
                KNeighborsClassifier, Random Forest Classifier, AdaBoost Classifier, Bagging Classifier, Extra Trees Classifier, Gradient Boosting Classifier, 
                XGBoost Classifier.
                Also Implemented Soft Voting Classifier and Stacking Classifier with estimators Multinomial NB, Random Forest, Extra Trees Classifier 
                Also tried out by applying Scaling (MinmaxScaling)
                Only consider 3000 most common words beacuse I get more accuracy and precision score

Evaluation: Assesses model performance using metrics such as accuracy, precision and confusion Matrix to ensure robustness and reliability.

Libraries Used:
               pandas: Data manipulation and analysis.
               nltk: Natural Language Toolkit for text processing.
               numpy: Numerical computing.
               sklearn: Machine learning algorithms and tools.
               seaborn: Data visualization.
               streamlit: Deployment and interactive web applications.

Results: The final model, Multinomial Naive Bayes, achieves an accuracy of 97.09% and a precision score of 1 on the test datasets, demonstrating its effectiveness in
         identifying spam messages. (Also achieving the same results with Voting Classifier)
         
