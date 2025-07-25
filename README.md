# Chat Intent Classification Pipeline

This repository contains an end‑to‑end pipeline for classifying user intents or chats/logs using an AI model (BERT).

## 📊 Project Highlights

- Fine-tuned BERT model on custom annotated chat intent data
- Precision, Recall, and F1 Score for each intent class: `[]`
- Interactive confusion matrix (via Plotly)
- Live intent prediction for new messages

**Technical Stack:**

* **Platforms:** GitHub, Hugging Face Hub, Streamlit Cloud
* **Languages:** Python
* **Databases:** MySQL
* **Libraries:** `transformers` (Hugging Face), `PyTorch`, `pandas`, `numpy`, `scikit-learn`, `sqlalchemy`, ``pymysql`, `plotly`, `streamlit`, `huggingface_hub`



## Table of Contents

* Project Overview
* 🌟 Key Features
* 📈 Pipeline Architecture
* 🚀 Live Demo
* 📊 Model Performance
* ⚙️ Setup and Local Execution
    * 1. Clone the Repository
    * 2. Create and Activate Virtual Environment
    * 3. Install Dependencies
    * 4. MySQL Database Setup
    * 5. Hugging Face Hub Authentication
    * 6. Run the Pipeline (Sequentially)
    * 7. Run Streamlit App Locally
* ☁️ Deployment
    * 8. Data Sources & Management
* 🔮 Future Enhancements
* 🙏 Credits & License



## Project Overview

### What does the project do?

This project implements a complete, end-to-end Machine Learning Operations (MLOps) pipeline for **Intent Classification**. It automatically processes raw text messages, fine-tunes a powerful **BERT (Bidirectional Encoder Representations from Transformers)** model to understand and categorize the underlying intent of these messages, and then deploys this model as an interactive web application for real-time predictions. The pipeline handles everything from data ingestion and cleaning, through secure storage in a MySQL database, to model training, evaluation, versioning on Hugging Face Hub, and public deployment on Streamlit Cloud.

### What does the project showcase?

* Demonstrates strong data handling, cleaning, feature engineering (through tokenization), and the ability to interpret model outputs (metrics, confusion matrices). It highlights understanding of how AI models consume and process data for business insights.

* Exhibits proficiency in data pipeline construction, SQL database management, data transformation, and presenting analytical insights (via Streamlit dashboards).

* Showcases expertise in deep learning model fine-tuning (BERT), MLOps principles (end-to-end pipeline, model versioning, deployment), efficient resource management (artifacts, checkpoints), and integration of various ML tools (Hugging Face, Streamlit, MySQL).

* Directly applies state-of-the-art NLP techniques for text classification using transformer models.


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

```mermaid
    A[Raw Data] --> B(ingest_clean.py);
    B --> C(load_to_mysql.py);
    C --> D[MySQL Database];
    D --> E(prepare_data.py);
    E --> F[Artifacts Folder];
    F --> G(finetune_bert.py);
    G --> H[BERT Output Folder];
    F & H --> I(export_bert_metrics.py);
    I --> J(push_model.py);
    J --> K[Hugging Face Hub];
    K --> L(app.py);
    L --> M[Streamlit Cloud Deployment];
    M --> N[Live Web Application];
    N --> O[User Interaction];





# 🚀 Live Demo
Experience the deployed application and test its intent classification capabilities in real-time:

👉 https://chat-intent-pipeline-6lzekhn44pl4ep6kklhdjk.streamlit.app/  [Live Streamlit App]


# 📊 Model Performance

## Intent Classification Model Training
The `finetune_bert.py` script orchestrates the training process, leveraging the Hugging Face Transformers library.

* Training Data: Custom dataset (details on preprocessing and tokenization are handled separately)
* Training Epochs: 3
* Batch Sizes: Train: 16, Evaluation: 32
* Framework: PyTorch with Hugging Face Trainer API

### Key Results (from Validation Set during training):
The model was evaluated on a held-out validation set after each epoch. The final performance metrics, based on the best model, are:

* Validation Loss: Approximately 0.0078
* Validation Accuracy: Approximately 99.88%

These results demonstrate the model's high accuracy and effectiveness in identifying user intents. The fine-tuned model checkpoint is committed to this repository and is located in the `artifacts/bert_intent_model/` directory.



## Test Set Metrics

The fine-tuned BERT model's performance on the held-out test set is summarized below. A full interactive confusion matrix is available in the live Streamlit dashboard for deeper analysis.

*Accuracy = 0.925
*Precision =0.920
*Recall = 0.930
*F1-Score = 0.925




# ⚙️ Setup and Local Execution

To set up and run this project locally, follow these steps:

## 1. Clone the Repository

* git clone https://github.com/farookahmedalishaik/chat-intent-pipeline.git
* cd chat-intent-pipeline

## 2. Create and Activate Virtual Environment

It's highly recommended to use a virtual environment.

* python -m venv venv
* On Windows: .\venv\Scripts\activate
* On macOS/Linux: source venv/bin/activate

## 3. Install Dependencies
* pip install -r requirements.txt

## 4. MySQL Database Setup
*Ensure you have a MySQL server running (e.g., via Docker, XAMPP, or a standalone installation).

*Create a database named `intent_db` and a user `intent_user` with password (or update `conn_str` in `load_to_mysql.py` and `prepare_data.py` with your credentials).

*Create the messages table:

* CREATE TABLE messages (
* id INT AUTO_INCREMENT PRIMARY KEY,
* text TEXT NOT NULL,
* label VARCHAR(255) NOT NULL,
* original_label VARCHAR(255)
* );



## 5. Hugging Face Hub Authentication
To push models and data to Hugging Face Hub, you need to authenticate locally.

* huggingface-cli login

Follow the prompts to paste your Hugging Face token (ensure it has write access for pushing).


## 6. Run the Pipeline (Sequentially)
Execute each Python script in the following order from the project root. This will ingest data, train the model, evaluate it, and push artifacts to Hugging Face Hub.

### Ingest and Clean Data:
* python ingest_clean.py

### Load Data to MySQL:
* python load_to_mysql.py

### Prepare Data for BERT (Creates/clears `artifacts/` folder):
* python prepare_data.py

### Fine-tune BERT Model (Creates `bert_output/` and updates `artifacts/bert_intent_model/`):
* python finetune_bert.py

### Export Model Metrics (Updates `artifacts/`):
* python export_bert_metrics.py

### Push Model & Artifacts to Hugging Face Hub:
* python push_model.py


## 7. Run Streamlit App Locally (for testing and optional)
To test the Streamlit dashboard on your local machine:

* streamlit run app.py
This will open the application in your web browser.



# ☁️ Deployment
The application is deployed on Streamlit Cloud, directly from this GitHub repository. **Continuous deployment** is enabled, meaning any push to the `main` branch will trigger an automatic redeployment.

For reliable access to Hugging Face models during deployment, a **read-only Hugging Face API token** is securely configured as a Streamlit Secret named `HF_TOKEN`. This token is not exposed in the codebase.




## Hugging Face Model Repository:
* Link to your Hugging Face model page something as : https://huggingface.co/farookahmedalishaik/intent-bert



# 8. Data Sources & Management
This project utilizes a publicly available dataset for training and evaluating the intent classification model downloaded from kaggle.

## Primary Dataset:
* Source: The core dataset for intent classification was acquired from **Kaggle**.
* Description: This dataset comprises customer interaction messages, each annotated with a specific intent label. It serves as the foundation for training the BERT model to understand and categorize diverse user queries.
* Kaggle Link: https://www.kaggle.com/code/mohamedchahed/customer-intent-classfication
* Local Storage: The raw dataset is stored within this repository at `data/raw_customer_interactions.csv`.




## Data Ingestion & Cleaning:
The `ingest_clean.py` script is responsible for loading the raw data, performing initial cleaning operations (e.g., handling missing values, standardizing text), and generating a cleaned version (`data/cleaned_all.csv`).

This cleaned data is then loaded into a **MySQL database** via `load_to_mysql.py` for structured storage and efficient retrieval. This database (`intent_db` with table `messages`) acts as the central data repository for the pipeline.




## Data Usage & Compliance:
* Licensing: Users should refer to the licensing terms specified on the original Kaggle dataset page for full details regarding usage rights. Most Kaggle datasets are under a permissive license (e.g., Creative Commons), allowing for non-commercial and often commercial use, but it's always best to verify directly from the source.

* Privacy: As the dataset is publicly available on Kaggle, it is generally considered anonymized or pre-processed. However, when working with any real-world customer interaction data, always prioritize data privacy (e.g., PII removal, secure storage, access control) in production environments. This project focuses on demonstrating the technical pipeline.


## 🔮 Future Enhancements

* **"Other" Intent Handling & Low Confidence Routing:** Implement a mechanism to classify user input as an "other" or "unclear" intent if it doesn't align with the model's trained intents. Additionally, if a prediction's confidence score falls below a predefined threshold, automatically route that message for human review or to the "other" intent for further investigation.

* **Sentiment Analysis Integration:** Implement a separate module to detect the sentiment (positive, negative, neutral) of the classified text messages. This would provide deeper insights into the user's emotional state in addition to their intent.

* **Advanced Evaluation:** Incorporate more detailed metrics like ROC curves, precision-recall curves, and per-class reports for a deeper understanding of model performance.

* **Data Versioning (DVC):** Implement Data Version Control (DVC) for datasets and models to track changes robustly and ensure reproducibility.

* **MLflow Integration:** Use MLflow for comprehensive experiment tracking, model registry, and improved reproducibility of training runs.

* **Real-time Inference API:** Explore deploying the model as a standalone API service (e.g., with FastAPI) for integration into other applications, offering more flexible real-time predictions.

* **Containerization (Docker):** Create Docker images for each pipeline component to ensure consistent environments across development, testing, and deployment.

* **User Feedback Loop:** Implement a mechanism to collect user feedback directly from the Streamlit app to facilitate model retraining with new, relevant data.

* **Automated Retraining:** Set up automated triggers (e.g., GitHub Actions) to retrain the model periodically or when new data becomes available in the MySQL database.


# 🙏 Credits & License

## Credits & Acknowledgements

A special thanks to the following for providing essential resources:

* Dataset: The customer intent classification dataset used in this project was sourced from Kaggle, provided by Mohamed Chahed.

* Dataset Link: https://www.kaggle.com/code/mohamedchahed/customer-intent-classfication

* Hugging Face Transformers: For providing the pre-trained BERT model and the robust transformers library, which is fundamental to this project's NLP capabilities.

* Streamlit: For enabling quick and easy deployment of interactive web applications for machine learning models.

* PyTorch: The underlying deep learning framework used for model training.



## License
This project is open-sourced under the **MIT License**. You are free to use, modify, and distribute this code, provided the original license and copyright notice are included.
The dataset used in this project is subject to its own licensing terms on Kaggle. Please refer to the original Kaggle dataset page for specific details regarding its usage rights.





1)Promotions & Discounts&sales &qureis related to promotions or discount 

Examples: “Do you have any promo codes?” “Is there a student discount?”

which month will get huge discounts..?

does prmotional offer apply to my kids

can i forward my discount coupon or promtion couopon to someone in my famiily

when does the promoiton offer ends

when doe sht discount ends

what are the discounts for vaious kinds 



2)Feedback / Suggestions / Thank You”

3)“Opening Hours / Business Policies”