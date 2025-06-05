from rest_framework.decorators import api_view
from rest_framework.response import Response
import joblib
import sys
import os
import json
import requests

# Add the Backend_Models directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from Backend_Models.NERModel.NER_RAGModel import extract_entities, get_conversational_response,replyFollowingQuerry

model = joblib.load("D:/projects/AI-Trip-Planner/New-Folder/travel_backend/api/intent_classifier_model.joblib")

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


@api_view(['GET'])
def get_place_image(request):
    place = request.GET.get('place', '')
    if not place:
        return Response({'error': 'Place parameter is required'}, status=400)
    print("Searching for place:", place)
    try:
        # First, search for the place to get its place_id using the new Places API
        search_url = f"https://places.googleapis.com/v1/places:searchText"
        headers = {
            'Content-Type': 'application/json',
            'X-Goog-Api-Key': 'AIzaSyAzdkr04QUofh6PvgTVHGeGvwIEpJulWLQ',
            'X-Goog-FieldMask': 'places.id,places.displayName,places.photos'
        }
        search_data = {
            'textQuery': place,
            'locationBias': {
                'circle': {
                    'center': {
                        'latitude': 10.8505,  # Kerala's approximate center
                        'longitude': 76.2711
                    },
                    'radius': 50000.0  # 50km radius
                }
            }
        }
        
        # print("Search URL:", search_url)
        search_response = requests.post(search_url, json=search_data, headers=headers)
        response_data = search_response.json()
        print("Search Response:", response_data)

        if not response_data.get('places') or len(response_data['places']) == 0:
            print("No results found in search response")
            return Response({'error': 'No results found'}, status=404)

        place_id = response_data['places'][0]['id']
        # print("Found place_id:", place_id)

        # Get the photo reference from the search response
        if not response_data['places'][0].get('photos'):
            print("No photos found in search response")
            return Response({'error': 'No photos found'}, status=404)

        photo_name = response_data['places'][0]['photos'][0]['name']
        
        # Return the photo URL using the new Places API
        image_url = f"https://places.googleapis.com/v1/{photo_name}/media?key=AIzaSyAzdkr04QUofh6PvgTVHGeGvwIEpJulWLQ&maxWidthPx=800"
        # print("Generated image URL:", image_url)
        return Response({'imageUrl': image_url})

    except Exception as e:
        print("Error occurred:", str(e))
        return Response({'error': str(e)}, status=500) 