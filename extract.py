import pandas as pd

# 1. Read your existing CSV file.
# The path now correctly includes the 'data' subfolder.
df = pd.read_csv("D:/Projects/chat-intent-pipeline/data/experimental.csv")

# 2. Get the counts of unique values in the 'intent' column.
intent_counts = df['intent'].value_counts()

# 3. Print the unique labels and their counts.
print("Unique labels and their counts:")
print(intent_counts)

