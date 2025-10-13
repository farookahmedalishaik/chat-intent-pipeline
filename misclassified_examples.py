import torch, pandas as pd, numpy as np
from config import TEST_DATA_FILE, LABEL_MAPPING_FILE, TEST_PREDS_FILE, ANALYSIS_MISCLASSIFIED_SAMPLE_FILE, ARTIFACTS_DIR
test = torch.load(TEST_DATA_FILE, weights_only=False)
preds = np.load(TEST_PREDS_FILE)
labels_df = pd.read_csv(LABEL_MAPPING_FILE)
labels = labels_df['label'].tolist()

y_true = test['labels'].cpu().numpy() if hasattr(test['labels'], 'cpu') else np.asarray(test['labels'])
texts = test.get('texts', [])
df = pd.DataFrame({'text': texts, 'true_id': y_true, 'pred_id': preds})
df['true'] = df['true_id'].apply(lambda x: labels[int(x)])
df['pred'] = df['pred_id'].apply(lambda x: labels[int(x)])
print("invoice -> pred bill examples:")
print(df[(df.true=='request_invoice') & (df.pred=='request_bill')].head(20))
print("\nbill -> pred invoice examples:")
print(df[(df.true=='request_bill') & (df.pred=='request_invoice')].head(20))
# Save for deeper manual review
df[(df.true=='request_invoice') & (df.pred=='request_bill')].to_csv(ARTIFACTS_DIR+"/inv_as_bill_sample.csv", index=False)
df[(df.true=='request_bill') & (df.pred=='request_invoice')].to_csv(ARTIFACTS_DIR+"/bill_as_inv_sample.csv", index=False)
