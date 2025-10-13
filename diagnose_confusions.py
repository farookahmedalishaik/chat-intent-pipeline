import pandas as pd
from config import TEST_CONFUSION_FILE
cm = pd.read_csv(TEST_CONFUSION_FILE, index_col=0)
print(cm.loc[['request_invoice','request_bill'], ['request_invoice','request_bill']])
print("\nTop rows that are predicted as bill but actual not bill:")
print(cm['request_bill'].sort_values(ascending=False).head(10))
