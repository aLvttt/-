import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[\W_]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

file_path = 'Sentiment140.csv'
data = pd.read_csv(file_path, encoding='ISO-8859-1', header=None, usecols=[0, 5], names=['sentiment', 'text'])

data['text'] = data['text'].apply(clean_text)

data['sentiment'] = data['sentiment'].map({0: 0, 4: 1})

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(train_data['text'])

train_sequences = tokenizer.texts_to_sequences(train_data['text'])
test_sequences = tokenizer.texts_to_sequences(test_data['text'])

max_length = 50
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding='post', truncating='post')
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post', truncating='post')

train_labels = train_data['sentiment'].values
test_labels = test_data['sentiment'].values

model = Sequential([
    Embedding(input_dim=10000, output_dim=64, input_length=max_length),
    Bidirectional(LSTM(64)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(train_padded, train_labels, epochs=2, batch_size=32, validation_data=(test_padded, test_labels))

loss, accuracy = model.evaluate(test_padded, test_labels)
print(f'Test Accuracy: {accuracy:.4f}')

predictions = model.predict(test_padded)
predicted_labels = (predictions > 0.5).astype(int).flatten()

print("Classification Report:")
print(classification_report(test_labels, predicted_labels))

print("Confusion Matrix:")
cm = confusion_matrix(test_labels, predicted_labels)
print(cm)

roc_auc = roc_auc_score(test_labels, predictions)
print(f'ROC AUC: {roc_auc:.4f}')

fpr, tpr, thresholds = roc_curve(test_labels, predictions)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

results = pd.DataFrame({'text': test_data['text'], 'predicted_sentiment': predicted_labels, 'actual_sentiment': test_labels})

results.to_csv('classification_results.csv', index=False)
print("Results have been saved to 'classification_results.csv'")
