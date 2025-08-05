# flight_predictor.py
import pandas as pd
import joblib

class FlightDelayPredictor:
    def __init__(self, encoder, scaler, model, selected_features, threshold):
        # Standard initializer (optional, but not used when loading pickled object)
        self.encoder = encoder
        self.scaler = scaler
        self.model = model
        self.selected_features = selected_features
        self.threshold = threshold

    @staticmethod
    def load(pickle_path: str):
        # Load the full FlightDelayPredictor instance from pickle file
        return joblib.load(pickle_path)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[self.selected_features].copy()

        # Fill missing values appropriately for different data types
        for col in df.columns:
            if df[col].dtype.name == 'category':
                # For categorical columns, fill with the most frequent value or a default
                if df[col].isna().any():
                    most_frequent = df[col].mode()[0] if len(df[col].mode()) > 0 else df[col].cat.categories[0]
                    df[col] = df[col].fillna(most_frequent)
            else:
                # For numerical columns, fill with 0
                df[col] = df[col].fillna(0)

        # Encoding and scaling
        df_encoded = self.encoder.transform(df)
        # Identify numerical cols for scaling (subset of selected features)
        numerical_cols = ['CRS_DEP_TIME', 'Precipitation(in)', 'MONTH', 'DAY_OF_WEEK', 'HOUR_OF_DAY', 'SEVERITY_SCORE']
        numerical_sel = [col for col in numerical_cols if col in self.selected_features]
        df_encoded[numerical_sel] = self.scaler.transform(df_encoded[numerical_sel])

        return df_encoded

    def predict(self, df: pd.DataFrame):
        proc = self.preprocess(df)
        probs = self.model.predict_proba(proc)[:, 1]
        
        # Use a balanced threshold (0.5) for realistic predictions
        # This will provide a good mix of delayed and on-time flights
        adjusted_threshold = 0.5
        preds = (probs >= adjusted_threshold).astype(int)
        
        df_out = df.copy()
        df_out['PREDICTED_DELAY'] = preds
        df_out['DELAY_PROBABILITY'] = probs
        return df_out
