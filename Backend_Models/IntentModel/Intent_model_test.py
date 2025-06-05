import joblib

# Load the trained model
model = joblib.load("intent_classifier_model.joblib")
print("okay till here")

# Example test prompts
test_prompts = [
    "I want to travel to Goa from Delhi on 10th June for 2 people.",
    "Will I find hotels in Manali?",
    "Plan a trip from Mumbai to Kerala in August.",
    "How much does a trip to Jaipur usually cost?"
]

# Predict intents
predicted_intents = model.predict(test_prompts)


# Print results
for prompt, intent in zip(test_prompts, predicted_intents):
    print(f"Prompt: {prompt}\n→ Predicted Intent: {intent}\n")
