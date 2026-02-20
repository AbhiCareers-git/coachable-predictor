import pandas as pd
import joblib
import tensorflow as tf

model = tf.keras.models.load_model("coachable_event_model.keras")
artifacts = joblib.load("preprocessing_artifacts.pkl")

ct = artifacts["column_transformer"]
rare_states = artifacts["rare_states"]
rare_vehicles = artifacts["rare_vehicles"]
required_cols = artifacts["required_columns"]

def predict_coachable_event(new_data_df, threshold=0.7):
    df = new_data_df.copy()

    df["group_level_3"] = df["group_level_3"].apply(
        lambda s: s[-1] if isinstance(s, str) else s
    )
    
    df["State"] = df["State"].replace(rare_states, "other")
    df["vehicle_type"] = df["vehicle_type"].replace(rare_vehicles, "Other")
    
    df = df[required_cols]
    
    x_trf = ct.transform(df)
    prob = model.predict(x_trf).flatten()[0]
    
    is_coachable = int(prob > threshold)
    
    return {"probability": float(prob), "coachable_flag": is_coachable}

def predict_batch_csv(df, threshold=0.7):
    original_df = df.copy()
    df["group_level_3"] = df["group_level_3"].apply(
        lambda s: s[-1] if isinstance(s, str) else s
    )
    df["State"] = df["State"].replace(rare_states, "other")
    df["vehicle_type"] = df["vehicle_type"].replace(rare_vehicles, "Other")
    df = df[required_cols]
    x_trf = ct.transform(df)
    pred_probs = model.predict(x_trf).flatten()
    original_df["predicted_probability"] = pred_probs
    original_df["coachable_flag"] = (pred_probs > threshold).astype(int)
    
    return original_df