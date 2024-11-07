Complaint Classification Model

This repository contains a complaint classification model built using Logistic Regression and Support Vector Machine (SVM). The model is designed to classify complaints based on their textual content into predefined categories. It uses techniques like TF-IDF vectorization and machine learning classifiers to predict the appropriate category.

Features
Text preprocessing of complaints to clean and standardize data.
Model training using Logistic Regression and SVM classifiers.
Text classification on new complaints using the trained model.
Web interface for submitting complaints and viewing predicted categories.
Prerequisites
Before running this project, ensure you have the following dependencies installed:

Python 3.9 or higher
scikit-learn for machine learning models
pandas for data handling
numpy for numerical operations
joblib or pickle for saving the trained models
Django for web interface 
matplotlib or seaborn for visualization 
tensorflow for neural network architecture development

Setup
1. Clone the Repository
Clone the repository to your local machine using Git:


git clone https://github.com/SArora04/IndiaAI-Hackathon.git

2. Install Dependencies
Install the dependencies using pip:
pip install -r requirements.txt

4. Training the Model
The model can be trained using the following script. It uses Logistic Regression and SVM to classify complaints based on their content.

cd bytebrigade
python manage.py runserver

The training script will:
Preprocess the complaint data (remove special characters, stop words, and apply TF-IDF) and classify the complaint as per the algorithm by resulting into category and subcategory.


5. Web Interface
A basic web interface is included to allow users to input complaints and view the predicted categories and subcategories.

Run the Web Server:

If you're using Django, you can run:

python manage.py runserver
