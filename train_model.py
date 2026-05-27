import numpy as np
print("Training started...")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


# features, labels ni import cheyyali
from extract_features import features, labels

# convert to numpy
X = np.array(features)
y = np.array(labels)

# encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# one hot encoding
y_categorical = to_categorical(y_encoded)

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42
)

# build model
model = Sequential()
model.add(Dense(256, input_shape=(40,), activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(y_categorical.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# train
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
model.save("emotion_model.h5")
print("Model saved successfully")
print("Training completed")

# evaluate
loss, acc = model.evaluate(X_test, y_test)
print("Accuracy:", acc)

