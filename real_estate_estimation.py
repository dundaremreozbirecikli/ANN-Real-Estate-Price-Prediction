import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.regularizers import l2
from keras.callbacks import EarlyStopping

class RealEstateMarketPrediction:
    def __init__(self):
        self.data = pd.read_csv("houses_Madrid.csv")
        self.model = None
        # Applying logarithmic transformation to buy price
        self.data['log_buy_price'] = np.log1p(self.data['buy_price'])
        
    def preprocess_data(self):
        drop_features = ['Unnamed: 0', 'id', 'title', 'latitude', 'longitude', 'portal', 'door', 'rent_price_by_area', 'buy_price']
        num_columns = self.data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_columns = self.data.select_dtypes(include=['object', 'bool']).columns.tolist()

        num_columns = [column for column in num_columns if column not in drop_features + ['log_buy_price']]
        categorical_columns = [column for column in categorical_columns if column not in drop_features]

        numeric_transform = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                            ('scaler', StandardScaler())])
        categorical_transform = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                                ('onehot', OneHotEncoder(handle_unknown='ignore'))])
        preprocess = ColumnTransformer(transformers=[('num', numeric_transform, num_columns),
                                                     ('cat', categorical_transform, categorical_columns)])

        dataset_features = self.data.drop(columns=drop_features)
        preprocessed_dataset = preprocess.fit_transform(dataset_features)

        svd = TruncatedSVD(n_components=100)
        preprocessed_dataset = svd.fit_transform(preprocessed_dataset)

        return preprocessed_dataset

    def create_sequential_model(self, input):
        model = Sequential([Input(shape=(input,)),Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
                            Dropout(0.2),Dense(32, activation='relu', kernel_regularizer=l2(0.001)),Dense(1)])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def model_train(self, x_train, y_train):
        self.model = self.create_sequential_model(x_train.shape[1])
        early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        self.model.fit(x_train, y_train, batch_size=64, callbacks=[early_stopping],validation_split=0.2, epochs=100)

    def model_evaluate(self, x_test, y_test):
        loss = self.model.evaluate(x_test, y_test)
        print(f"Test Loss: {loss}")

    def predict(self, x):
        return self.model.predict(x)

    def plot_predictions_real(self, X_test, y_test):
        predictions = self.predict(X_test)
        plt.figure(figsize=(10, 6))
        plt.scatter(np.expm1(y_test), np.expm1(predictions), alpha=0.3)
        plt.title('Real Prices vs Predicted Prices')
        plt.xlabel('Real Prices')
        plt.ylabel('Predicted Prices')
        plt.plot([np.expm1(y_test).min(), np.expm1(y_test).max()], [np.expm1(y_test).min(), np.expm1(y_test).max()], 'k--')
        plt.show()


predictor = RealEstateMarketPrediction()

preprocessed_data = predictor.preprocess_data()
prediction_object = predictor.data['log_buy_price']

X_train, X_test, y_train, y_test = train_test_split(preprocessed_data, prediction_object, test_size=0.25, random_state=6)

predictor.model_train(X_train, y_train)
predictor.model_evaluate(X_test, y_test)
predictor.plot_predictions_real(X_test, y_test)
predictions = predictor.predict(X_test)

# Reverting log transformation for both predicted and real property prices
actual_prices = np.expm1(y_test)
predicted_prices = np.expm1(predictions)

for i in range(10):
    print(f"Predicted price: {round(predicted_prices[i][0])}, Real price: {round(actual_prices.iloc[i])}")

