import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

def generate_training_data(num_samples=1000):
    """
    Generates synthetic training data based on Russell's Circumplex Model of Affect
    mapping to musical constraints.
    """
    print("Generating music theory mapping rules...")
    
    # Random [Valence, Arousal] from -1.0 to 1.0
    X = np.random.uniform(-1.0, 1.0, (num_samples, 2))
    y = np.zeros((num_samples, 3)) # [BPM, Key, Density]
    
    for i in range(num_samples):
        valence, arousal = X[i, 0], X[i, 1]
        
        # 1. BPM (Driven by Arousal): -1.0 -> 60 BPM, 1.0 -> 180 BPM
        bpm = 120 + (arousal * 60)
        
        # 2. Key Signature (Driven by Valence): > 0 is Major (mapped higher), < 0 is Minor
        key = 1.0 if valence > 0 else 0.0 
        
        # 3. Note Density (Driven by both): 0.0 (Sparse) to 1.0 (Dense)
        # High energy and positive emotion generally leads to denser notes
        density = (valence + arousal + 2) / 4.0 
        
        y[i] = [bpm, key, density]
        
    return X, y

def main():
    print("Initializing Random Forest Regressor...")
    # Initialize the model with 100 decision trees
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Get our training data
    X_train, y_train = generate_training_data(num_samples=5000)
    
    print("Training the Mapper...")
    rf_model.fit(X_train, y_train)
    
    print("Training Complete!")
    
    # Let's test it to prove it works
    print("\n--- Testing the Bridge ---")
    
    # Test Case 1: "Happy" (High Valence, High Arousal)
    happy_coords = np.array([[0.8, 0.6]])
    happy_music = rf_model.predict(happy_coords)
    print(f"Input 'Happy' [0.8, 0.6] -> Output [BPM, Key, Density]: {happy_music[0]}")
    
    # Test Case 2: "Sad" (Low Valence, Low Arousal)
    sad_coords = np.array([[-0.8, -0.6]])
    sad_music = rf_model.predict(sad_coords)
    print(f"Input 'Sad' [-0.8, -0.6] -> Output [BPM, Key, Density]: {sad_music[0]}")
    
    # Save the trained model so the rest of your pipeline can use it
    joblib.dump(rf_model, 'random_forest_mapper.pkl')
    print("\nModel saved as 'random_forest_mapper.pkl'")

if __name__ == "__main__":
    main()