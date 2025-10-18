# Chat Intent Classification Pipeline

This repository contains an end‑to‑end pipeline for classifying user intents or chats/logs using an AI model (BERT).

## 📊 Project Highlights

- Fine-tuned BERT model on custom annotated chat intent data
- Precision, Recall, and F1 Score for each intent class
- Interactive confusion matrix for error analysis
- Live, real-time intent prediction through a deployed web application

**Technical Stack:**

* **Platforms:** GitHub, Hugging Face Hub, Streamlit Cloud
* **Languages:** Python
* **Databases:** MySQL
* **Libraries:** `transformers` (Hugging Face), `PyTorch`, `spaCy` `Presidio` `pandas`, `numpy`, `scikit-learn`, `sqlalchemy`, `pymysql`, `plotly`, `streamlit`, `huggingface_hub`


👉 https://chat-intent-pipeline-qvnch3q6hnnrknrjdnurdk.streamlit.app/ **[View Live Streamlit App]**


## Table of Contents

* Project Overview
* Key Features
* Pipeline Architecture
* Live Demo
* Model Performance
* Data Sources & Management
* Setup and Local Execution
    * 1. Prerequisites
    * 2. Clone the Repository
    * 3. Configure Environment
    * 4. Install Dependencies
    * 5. Run the Pipeline
    * 6. Run Streamlit App Locally
* Deployment
* Future Enhancements
* Credits & License



## Project Overview

### What does the project do?

This project implements a complete, end-to-end Machine Learning Operations (MLOps)/ AI Data Analyst pipeline for **Intent Classification**. It automatically processes text messages, fine-tunes a powerful **BERT (Bidirectional Encoder Representations from Transformers)** model to understand and categorize the underlying intent of these messages, and then deploys this model as an interactive web application for real-time predictions. The pipeline handles everything from data ingestion and cleaning, through secure storage in a **MySQL** database, to model training, evaluation, versioning on **Hugging Face Hub**, and public deployment on **Streamlit Cloud**.

### What does the project showcase?

* Demonstrates strong data handling, cleaning, feature engineering (through tokenization), and the ability to interpret model outputs (metrics, confusion matrices). It highlights understanding of how AI models consume and process data for business insights.

* Exhibits proficiency in data pipeline construction, SQL database management, data transformation, and presenting analytical insights (via Streamlit dashboards).

* Showcases expertise in deep learning model fine-tuning (BERT), MLOps principles (end-to-end pipeline, model versioning, deployment), efficient resource management (artifacts, checkpoints), and integration of various ML tools (Hugging Face, Streamlit, MySQL).

* Directly applies state of the art NLP techniques for text classification using transformer models.


### Summary

**This project builds a robust MLOps pipeline to automatically classify text message intents using a fine-tuned BERT model, making real-time predictions accessible via a deployed Streamlit web application.** It's a comprehensive demonstration of deep learning, data engineering, and seamless deployment in action.



## 🌟 Key Features

This project isn't just about building a model; it's about building a robust, repeatable, and insightful AI-driven data solution. Here's what makes this pipeline stand out:

* **Intelligent Intent Classification with BERT:** At its core, this pipeline employs a fine-tuned BERT-base-uncased model, a state-of-the-art transformer architecture, to accurately understand and categorize the underlying intent of diverse text messages.Enabling sophisticated understanding of nuanced customer queries.

    * Demonstrates deep learning model application, understanding of complex NLP embeddings, and the ability to leverage pre-trained large language models for specific business problems.


* **Automated Data Pipeline & Management:** From raw csv inputs, the system performs automated data ingestion, cleaning, and structured storage in a MySQL database. This ensures data quality and accessibility for downstream tasks.

    * Highlights strong ETL (Extract, Transform, Load) capabilities, SQL proficiency, and the establishment of a reliable data source for analytical insights and model training.


* **Comprehensive Data Preparation for ML:** The pipeline intelligently splits the dataset into stratified training, validation, and test sets, ensuring balanced representation of all intent classes. It also handles label encoding and BERT-specific tokenization, transforming raw text into model-ready tensors.

    * Showcases meticulous attention to data quality for model performance, understanding of data leakage prevention, and expertise in preparing complex unstructured data for deep learning.


* **Rigorous Model Evaluation & Insights:** Beyond just accuracy, the project includes automated calculation and visualization of critical model metrics on an unseen test set, including precision, recall, F1-score, and a detailed confusion matrix. These results are pushed directly to Hugging Face Hub for transparency.

    * Emphasizes strong analytical skills in model performance interpretation, identifying areas for model improvement, and understanding classification biases through visual tools like the confusion matrix.


* **Reproducible MLOps Workflow:** The entire process, from data processing to model deployment, is encapsulated in modular Python scripts, enabling reproducible experimentation and seamless updates. Model weights, tokenizer, and evaluation reports are systematically versioned on Hugging Face Hub.

    * Demonstrates practical application of MLOps principles, emphasis on reusability, version control for artifacts, and establishing a single source of truth for model assets.


* **Interactive Real-Time Prediction Dashboard:** A user-friendly web application, built with Streamlit and deployed on Streamlit Cloud, allows anyone to input a message and instantly receive the predicted intent. This dashboard also visually presents the model's test-set metrics and confusion matrix.

    * Highlights ability to productize ML models, build intuitive user interfaces, and effectively communicate complex AI insights to non-technical stakeholders.







## 📈 Pipeline Architecture

The project follows a sequential data and model flow, ensuring modularity and reproducibility:

1. Data Ingestion & Cleaning (ingest_clean.py)

Purpose: Reads raw data, cleans it, normalizes text using preprocess.py (spaCy/Presidio), extracts placeholders, and saves the cleaned dataset (cleaned_all.csv) with mappings.

2. Database Loading (load_to_mysql.py)

Purpose: Loads cleaned data (cleaned_all.csv) into MySQL, using hash-based deduplication on raw text for data integrity.

3. Data Preparation for Model (prepare_data.py)

Purpose: Loads data (DB preferred), creates model-ready text (text_model) with feature tokens, encodes labels, performs group-aware train/val/test splits, calculates class weights, tokenizes text, and saves final datasets as .pt files and audit snapshots as .csv.

4. Model Fine-tuning (finetune_bert.py)

Purpose: Loads prepared data and base BERT model, fine-tunes on training data using Trainer (handles class imbalance), evaluates on validation data, calculates class-specific confidence thresholds (min 80%), and saves the best model locally.

5. Final Model Evaluation (export_bert_metrics.py)

Purpose: Loads the best model and test data (.pt), runs predictions on unseen data, and generates final metrics, confusion matrix, and prediction files.

6. Error Analysis (error_analysis_clean.py, analysis/export_top_confusions.py)

Purpose: Uses test predictions to identify low-performing classes and exports misclassified examples for review.

7. Artifact & Model Pushing (push_data_artifacts.py, push_model.py, push_evaluation_artifacts.py)

Purpose: Uploads key artifacts (prepared data, model, thresholds, metrics) to dedicated Hugging Face Hub repositories for central storage and versioning.

8. Sanity Checking (run_all_sanity_checks.py)

Purpose: Performs crucial QA checks (e.g., data overlap, collisions) after data preparation and evaluation to detect potential issues like leakage.

9. Application Serving (app.py)

Purpose: Runs the Streamlit dashboard, loading the model/artifacts from Hugging Face Hub, providing a UI for predictions (with confidence thresholding for "other"), displaying results/metrics, and logging predictions.

10. Supporting Files

* config.py: Centralizes project configurations (paths, hyperparameters, credentials, Hugging Face repo IDs).
* preprocess.py: Handles text cleaning and entity/placeholder replacement entities with using spaCy and Presidio.
* utils.py: Provides shared helper functions loading artifacts/models from (cloud loading, DB connection).
* .env: Stores sensitive secrets (API keys, passwords).
* requirements.txt: Lists project dependencies to run the project.



# 🚀 Live Demo
Experience the deployed application and test its intent classification capabilities in real-time:

👉 https://chat-intent-pipeline-qvnch3q6hnnrknrjdnurdk.streamlit.app/ [Live Streamlit App]



# 📊 Model Performance

## Intent Classification Model Training
The `finetune_bert.py` script orchestrates the training process, leveraging the Hugging Face Transformers library.

* Training Data: Preprocessed and tokenizded from the central MYSQL database
* Training Epochs: 6 (configurable in config.py)
* Batch Sizes: Train: 16, Evaluation: 32
* Framework: PyTorch with Hugging Face Trainer API

### Results (from Validation Set during training):
The model was evaluated on a held-out validation set after each epoch. The final performance metrics, based on the best model, are:

* Validation Loss: Approximately 0.0078
* Validation Accuracy: Approximately 99.88%

Note: These values might vary with each training run.



## Test Set Metrics

The fine-tuned BERT model's performance on the unseen test set is summarized below.A full interactive confusion matrix is available in the live Streamlit dashboard.

* Accuracy = 0.98+
* Precision =0.98+
* Recall = 0.98+
* F1-Score = 0.98+


# 🗄️ Data Sources & Management
The intent classification model was trained and evaluated using a publicly available dataset from Kaggle, which was then merged with custom-generated synthetic data. This synthetic data includes a mix of standard English, common typographical errors, and slang to enhance the model's robustness.

## Primary Dataset:
* Source: The core dataset for intent classification was acquired from **Kaggle**.
* Description: This dataset comprises customer interaction messages, each annotated with a specific intent label. It serves as the foundation for training the BERT model to understand and categorize diverse user queries.
* Kaggle Link: https://www.kaggle.com/code/mohamedchahed/customer-intent-classfication
* Local Storage: The raw dataset is stored within this repository at `data/raw_customer_interactions.csv`.




## Data Ingestion & Cleaning:
The `ingest_clean.py` script loads raw data, performs masking and clenaing using preprocess.py and saves a cleaned version to (`data/cleaned_all.csv`).

This cleaned data is then loaded into a **MySQL database** via `load_to_mysql.py` for structured storage and efficient retrieval. This database (`intent_db` with table `messages`) acts as the central data repository for the pipeline.



# ⚙️ Setup and Local Execution

To set up and run this project locally, follow these steps:


## 1. Prerequisites

* Python 3.8+
* Git
* MYSQL server like MYSQL Workbench

## 2. Clone the Repository

* git clone https://github.com/farookahmedalishaik/chat-intent-pipeline.git
* cd chat-intent-pipeline

## 3. Configure Environment

Create .env file in project root to hold secret credentials. environment.


## 4. Install Dependencies

It is highly recommended to use a virtual environment.

* python -m venv venv (# Create a virtual environment)

* .\venv\Scripts\activate (Activate it on Windows:)

* pip install -r requirements.txt (# Install required packages)

* python -m spacy download en_core_web_sm (# Download the spaCy model for preprocessing)



## 5. Run the Pipeline (Sequentially)

Make sure MYSQL server is running and have created a database like intent_db. The python scripts will create teh necessary tables automatically

### 1. Ingest ,clean and preproces raw data:
* python ingest_clean.py

### 2. Load Data to MySQL database:
* python load_to_mysql.py

### 3. Prepare and tokenzide data for BERT , saving to `artifacts/`:
* python prepare_data.py

### 4. Push Data Artifacts to Hugging Face Hub:
* push_data_artifacts.py

### 5. Fine-tune BERT Model (Creates `bert_output/` and updates `artifacts/bert_intent_model/`):
* python finetune_bert.py

### 6. Push the final Model & Artifacts to Hugging Face Hub:
* python push_model.py

### 7. Run Validation Dashboard
* python validation_dashboard.py

### 8. Evaluate the model and exports metrics (Updates `artifacts/`) (Test Set):
* python export_bert_metrics.py

### 9. Push Evaluation Metrics to Data Repository
* python push_evaluation_metrics.py

### 10. Targeted Error Analysis & Confusion Pair Analysis
* python error_analysis.py

* python analysis_export_top_confusions.py

### 11. Run All Sanity Checks
* python run_all_santity_cheks.py

## 12. Run Streamlit App Locally (for testing local deployment)
To test the Streamlit dashboard on your local machine:

* streamlit run app.py



# ☁️ Deployment
The application is deployed on Streamlit Cloud, directly from this GitHub repository. **Continuous deployment** is enabled, meaning any push to the `main` branch will trigger an automatic redeployment.

For the deployed app to function, the following are configures as Streamlit Secrets:

* HF_TOKEN: A read only Hugging Face API token

* MYSQL_USER, MYSQL_PASSWORD, MYSQL_HOST, MYSQL_DB: Credntials for the MYSQL database.



## Hugging Face Model Repository:
* Link to your Hugging Face model page something as : https://huggingface.co/farookahmedalishaik/intent-bert






## 🔮 Future Enhancements

* **Sentiment Analysis Integration:** Implement a separate module to detect the sentiment (positive, negative, neutral) of the classified text messages. This would provide deeper insights into the user's emotional state in addition to their intent.

* **Advanced Evaluation:** Incorporate more detailed metrics like ROC curves, precision-recall curves, and per-class reports for a deeper understanding of model performance.

* **Data Versioning (DVC):** Implement Data Version Control (DVC) for datasets and models to track changes robustly and ensure reproducibility.

* **MLflow Integration:** Use MLflow for comprehensive experiment tracking, model registry, and improved reproducibility of training runs.

* **Real-time Inference API:** Explore deploying the model as a standalone API service (e.g., with FastAPI) for integration into other applications, offering more flexible real-time predictions.

* **Containerization (Docker):** Create Docker images for each pipeline component to ensure consistent environments across development, testing, and deployment.

* **User Feedback Loop:** Implement a mechanism to collect user feedback directly from the Streamlit app to facilitate model retraining with new, relevant data.

* **Automated Retraining:** Set up automated triggers (e.g., GitHub Actions) to retrain the model periodically or when new data becomes available in the MySQL database.


# 🙏 Credits

## Credits & Acknowledgements

A special thanks to the following for providing essential resources:

* Dataset: The customer intent classification dataset used in this project was sourced from Kaggle, provided by Mohamed Chahed.

* Dataset Link: https://www.kaggle.com/code/mohamedchahed/customer-intent-classfication

* Hugging Face Transformers: For providing the pre-trained BERT model and the robust transformers library, which is fundamental to this project's NLP capabilities.

* Streamlit: For enabling quick and easy deployment of interactive web applications for machine learning models.

* PyTorch: The underlying deep learning framework used for model training.
