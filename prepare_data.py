# prepare_data.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer
import torch
from sqlalchemy import create_engine, text
import shutil

# Database connection

# database host is from an environment variable.
# If the environment variable DB_HOST is not set like when running locally outside Docker it will default to 'localhost'.
# But When running in Docker, we will explicitly set DB_HOST to 'host.docker.internal'.
DB_HOST = os.environ.get("DB_HOST", "localhost")

# connection string with respective host
conn_str = f"mysql+pymysql://intent_user:password:intent_db@{DB_HOST}:3306/intent_db"
engine = create_engine(conn_str)


# Delete and recreate artifacts folder if there is any
if os.path.exists("artifacts"):
    shutil.rmtree("artifacts")
    print(f" 'artifacts' folder deleted.")
os.makedirs("artifacts")
print(f" 'artifacts' folder created")


# Load data from MySQL database
print("Loading data from MySQL database")
with engine.connect() as conn:
    result = conn.execute(text("SELECT text, label FROM messages"))
    data = result.fetchall()
    df = pd.DataFrame(data, columns = result.keys())
print(f"Loaded {len(df)} rows from MySQL for data preparation")


print(f"Unique labels in dataframe after loading from mysql: {df['label'].unique().tolist()}")
print(f"Count of unique labels: {len(df['label'].unique())}")



# Using only 'text' and 'label'
texts = df["text"].tolist()
labels = df["label"].tolist()

# Encode labels to numeric form
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Save label mappings
pd.DataFrame({
    "label": label_encoder.classes_,
    "id": range(len(label_encoder.classes_))
}).to_csv(os.path.join("artifacts", "label_mapping.csv"), index = False)

# Train/validation/test split = (80/10/10)
X_temp, X_test, y_temp, y_test = train_test_split(
    texts, encoded_labels, test_size = 0.1, random_state = 42, stratify = encoded_labels
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size = 0.111, random_state = 42, stratify = y_temp
)

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Function to tokenize texts
def tokenize_texts(texts):
    return tokenizer(
        texts,
        padding = True,
        truncation = True,
        return_tensors = "pt",
        max_length = 128
    )

# Tokenize each split
train_encodings = tokenize_texts(X_train)
val_encodings = tokenize_texts(X_val)
test_encodings = tokenize_texts(X_test)

# Save processed datasets as torch tensors
torch.save({
    "input_ids": train_encodings["input_ids"],
    "attention_mask": train_encodings["attention_mask"],
    "labels": torch.tensor(y_train)
}, os.path.join("artifacts", "train_data.pt"))

torch.save({
    "input_ids": val_encodings["input_ids"],
    "attention_mask": val_encodings["attention_mask"],
    "labels": torch.tensor(y_val)
}, os.path.join("artifacts", "val_data.pt"))

torch.save({
    "input_ids": test_encodings["input_ids"],
    "attention_mask": test_encodings["attention_mask"],
    "labels": torch.tensor(y_test)
}, os.path.join("artifacts", "test_data.pt"))

print("Data preparation is completed. And Tensors are saved to 'artifacts/' ")