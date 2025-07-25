# prepare_data.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer
import torch
from sqlalchemy import create_engine, text # Added for database 
import shutil # to get shutil.rmtree (!!permanently deletes the folder and its contents!!)

# Database connection
conn_str = "mysql+pymysql://intent_user:password:intent_db@localhost:3306/intent_db"
engine = create_engine(conn_str)


# Delete and recreate artifacts folder if there is any
#'artifacts' is the folder
if os.path.exists("artifacts"):
    shutil.rmtree("artifacts")
    print(f" 'artifacts' folder deleted.")
os.makedirs("artifacts")
print(f" 'artifacts' folder created")


# Load data from MySQL database
print("Loading data from MySQL database")
with engine.connect() as conn:
    result = conn.execute(text("SELECT text, label FROM messages")) #'text()' for a raw SQL query
    data = result.fetchall() # Fetch all rows and convert to a list of dictionaries
    df = pd.DataFrame(data, columns=result.keys()) # Convert list of tuples/rows to dataframe
print(f"Loaded {len(df)} rows from MySQL for data preparation")


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
}).to_csv(os.path.join("artifacts", "label_mapping.csv"), index = False) # Use os.path.join

# Train/validation/test split (80/10/10)
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
}, os.path.join("artifacts", "train_data.pt")) #Use os.path.join

torch.save({
    "input_ids": val_encodings["input_ids"],
    "attention_mask": val_encodings["attention_mask"],
    "labels": torch.tensor(y_val)
}, os.path.join("artifacts", "val_data.pt")) # Use os.path.join

torch.save({
    "input_ids": test_encodings["input_ids"],
    "attention_mask": test_encodings["attention_mask"],
    "labels": torch.tensor(y_test)
}, os.path.join("artifacts", "test_data.pt")) #Use os.path.join

print("Data preparation is completed. And Tensors are saved to 'artifacts/'")