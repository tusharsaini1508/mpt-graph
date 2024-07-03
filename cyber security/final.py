import numpy as np
import pandas as pd
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras import layers
from tensorflow import keras
import matplotlib.pyplot as plt

# Load datasets
for dirname, _, filenames in os.walk('E:/SHALI NEW IMPLEMENTATION/Urvasi_Implementation/nsl-kdd'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train = pd.read_csv('nsl-kdd/KDDTrain+_20Percent.csv')
test = pd.read_csv('nsl-kdd/KDDTestdata.csv')

# Define columns
columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
           'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
           'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
           'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
           'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
           'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
           'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
           'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack', 'level']

train.columns = columns
test.columns = columns

# Add binary attack labels
train['is_attacked'] = (train.attack != 'normal').astype(int)
test['is_attacked'] = (test.attack != 'normal').astype(int)

# Combine train and test data
combined_data = pd.concat([train, test])
combined_data.reset_index(drop=True, inplace=True)

# OneHotEncode categorical features
features_to_encode = ['protocol_type', 'service', 'flag']
encoder = OneHotEncoder(sparse_output=False)
encoded_features = encoder.fit_transform(combined_data[features_to_encode])

encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(features_to_encode))
combined_data = combined_data.drop(columns=features_to_encode).reset_index(drop=True)
combined_data = pd.concat([combined_data, encoded_df], axis=1)

# Train/Test Split
train_cnt = train.shape
test_cnt = test.shape
train_data = combined_data[:train_cnt[0]]
test_data = combined_data[train_cnt[0]:]

X_train = train_data.drop(columns=['attack', 'level', 'is_attacked'])
y_train = train_data['is_attacked']
X_test = test_data.drop(columns=['attack', 'level', 'is_attacked'])
y_test = test_data['is_attacked']

# Define MLPNN model
def MLPNN(n_inputs):
    inputs = keras.Input(shape=(n_inputs,), name="input")
    batch = layers.BatchNormalization()(inputs)
    layer1 = layers.Dense(256, activation="sigmoid", name="dense1")(batch)
    drop = layers.Dropout(rate=0.5)(layer1)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(drop)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

MLPNN_model = MLPNN(X_train.shape[1])
MLPNN_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=1, restore_best_weights=True)

# Train MLPNN model
history = MLPNN_model.fit(X_train, y_train, batch_size=512, epochs=400, callbacks=[callback], validation_data=(X_test, y_test))

# Evaluate MLPNN model
MLPNN_predictions = MLPNN_model.predict(X_test, batch_size=512)
MLPNN_binary_predictions = (MLPNN_predictions > 0.5).astype(int)

accuracy = accuracy_score(y_test, MLPNN_binary_predictions)
precision = precision_score(y_test, MLPNN_binary_predictions)
recall = recall_score(y_test, MLPNN_binary_predictions)
f1 = f1_score(y_test, MLPNN_binary_predictions)

print(f"MLPNN Model - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

# Plotting training history
plt.figure(figsize=(12, 8))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

plt.figure(figsize=(12, 8))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# SVM and RandomForest for binary classification
binary_train_X, binary_val_X, binary_train_y, binary_val_y = train_test_split(X_train, y_train, test_size=0.6)
binary_model_svm = SVC()
binary_model_svm.fit(binary_train_X, binary_train_y)
binary_predictions_svm = binary_model_svm.predict(binary_val_X)

binary_model_rf = RandomForestClassifier()
binary_model_rf.fit(binary_train_X, binary_train_y)
binary_predictions_rf = binary_model_rf.predict(binary_val_X)

# Evaluate SVM model
accuracy_svm = accuracy_score(binary_val_y, binary_predictions_svm)
precision_svm = precision_score(binary_val_y, binary_predictions_svm)
recall_svm = recall_score(binary_val_y, binary_predictions_svm)
f1_svm = f1_score(binary_val_y, binary_predictions_svm)

print(f"SVM Model - Accuracy: {accuracy_svm}, Precision: {precision_svm}, Recall: {recall_svm}, F1 Score: {f1_svm}")

# Evaluate RandomForest model
accuracy_rf = accuracy_score(binary_val_y, binary_predictions_rf)
precision_rf = precision_score(binary_val_y, binary_predictions_rf)
recall_rf = recall_score(binary_val_y, binary_predictions_rf)
f1_rf = f1_score(binary_val_y, binary_predictions_rf)

print(f"RandomForest Model - Accuracy: {accuracy_rf}, Precision: {precision_rf}, Recall: {recall_rf}, F1 Score: {f1_rf}")
