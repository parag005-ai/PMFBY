"""Check trained model info."""
import pickle

with open('models/trained/yield_model.pkl', 'rb') as f:
    data = pickle.load(f)

print('MODEL INFO:')
print(f"  Model: {data['model_name']}")
print(f"  Training samples: {data['training_samples']}")
print(f"  Trained: {data['trained_on']}")
print()
print('PERFORMANCE METRICS:')
print(f"  MAE: {data['metrics']['mae']:.0f} kg/ha")
print(f"  RMSE: {data['metrics']['rmse']:.0f} kg/ha")
print(f"  R2: {data['metrics']['r2']:.3f}")
print()
print('FEATURES:')
for f in data['feature_cols']:
    print(f"  - {f}")
