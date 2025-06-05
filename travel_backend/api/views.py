from rest_framework.decorators import api_view
from rest_framework.response import Response
import joblib
import sys
import os
import json

# Add the Backend_Models directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from Backend_Models.NERModel.NER_RAGModel import extract_entities, get_conversational_response,replyFollowingQuerry

model = joblib.load("D:/projects/AI-Trip-Planner/travel_backend/api/intent_classifier_model.joblib")

@api_view(['POST'])
def process_query(request):
    data = request.data
    print(data)
    query = data.get('query', '')
    flag = data.get('flag','')
    context = data.get('context','')
    print("Received from frontend:", query)
    response_data = {}
    entitiy = {}  # Initialize entitiy variable

    if(flag == True):
        res = replyFollowingQuerry(context,query)
        print("Raw response:", res)
        # Clean the response string
        res = res.replace('```json', '').replace('```', '').strip()
        try:
            entitiy = json.loads(res)
        except json.JSONDecodeError as e:
            print("Failed to parse JSON:", e)
            entitiy = {}  # Set empty dict if parsing fails
        response_data = {
            "intent": "trip_info",
            "entities": entitiy
        }
        return Response(response_data)
    print(type(flag))
    print(flag)
    

    # Get intent from the classifier
    predicted_intent = model.predict([query])[0]
    # print("Predicted intent:", predicted_intent)
    print(predicted_intent)

    
    if predicted_intent == 'trip_info':
        # Extract entities for trip planning
        entities = extract_entities(query)
        print("Raw entities:", entities)
        # Clean the entities string
        entities = entities.replace('```json', '').replace('```', '').strip("` \n")

        try:
            entities = json.loads(entities)
        except json.JSONDecodeError as e:
            print("Failed to parse JSON:", e)
            entities = {}
        
        response_data = {
            "intent": "trip_info",
            "entities": entities
        }
    elif predicted_intent == 'general_query':
        # Get conversational response
        response = get_conversational_response(query)
        response_data = {
            "intent": "general_query",
            "response": response
        }
    else:
        response_data = {
            "intent": "unknown",
            "message": "Could not determine the intent of your query"
        }

    return Response(response_data)
