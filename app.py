from fastapi import FastAPI, File, UploadFile, HTTPException, Form,Request
from pydantic import BaseModel
from typing import List, Optional, Dict ,  Union
from pathlib import Path
import os
import tempfile
import shutil
import re
import httpx
import uuid
from utils.wisperapi import WhisperAPI
from utils.emotion_analyzer import EmotionAnalyzer
from utils.bag_of_words import BagOfWordsFilter
import json
import assemblyai as aai
from openai import OpenAI
from dotenv import load_dotenv
from utils.utils import crop_video_by_time, extract_human_top_left, process_video_segment , transcription_to_qa, convert_to_mp3,extract_human_responses,save_uploaded_file,process_transcription,prioritize_facial_expression
import torch
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
# OpenAI API Configuration
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY1"))
# Initialize AssemblyAI settings
aai.settings.api_key = os.getenv("assembly_API_KEY")
# Create FastAPI instance
app = FastAPI()

load_dotenv()

# Paths for temporary storage
base_dir = Path("./temp")
video_folder = base_dir / "videos"
audio_folder = base_dir / "audios"
os.makedirs(video_folder, exist_ok=True)
os.makedirs(audio_folder, exist_ok=True)

# Path to the custom sentiments file
custom_sentiments_file = base_dir / "custom_sentiments.json"

# Ensure the file exists
if not custom_sentiments_file.exists():
    with open(custom_sentiments_file, "w") as f:
        json.dump({}, f)
# Initialize APIs
whisper_api = WhisperAPI()
emotion_analyzer = EmotionAnalyzer()

# Pydantic Models
# New models for the endpoint
class NewSentimentResponse(BaseModel):
    timestamp_start: Optional[str]
    timestamp_end: Optional[str]
    question: str
    answer: str
    sentiment: str
# Define the models used for data validation
class SentimentRequest(BaseModel):
    id: str
    transcription: list

class CombinedInput(BaseModel):
    sentiment_data: dict
    facial_data: dict


class NewSentimentData(BaseModel):
    id: str
    responses_with_sentiment: List[NewSentimentResponse]


class NewFacialExpressionResult(BaseModel):
    start_time: Union[str, float]
    end_time: Union[str, float]
    expressions: Optional[str] = None
    error: Optional[str] = None


class NewFacialData(BaseModel):
    expressions: List[NewFacialExpressionResult]


class NewCombinedInput(BaseModel):
    sentiment_data: NewSentimentData
    facial_data: NewFacialData
    
class QAPair(BaseModel):
    bot: Optional[str]
    human: Optional[str]
    timestamp_start: Optional[str] = ""  # New field for start timestamp
    timestamp_end: Optional[str] = ""    # New field for end timestamp
class SentimentRequest(BaseModel):
    id: Optional[str]
    transcription: List[QAPair]
class SentimentInput(BaseModel):
    id: str
    responses_with_sentiment: list

class ExpressionsInput(BaseModel):
    expressions: str

class CombinedInput(BaseModel):
    sentiment_data: SentimentInput
    facial_data: ExpressionsInput
    
class HumanWordCheckRequest(BaseModel):
    id: str
    transcription: List[QAPair]
    words_to_check: List[str]

class HumanWordCheckResponse(BaseModel):
    id: str
    words_found: Dict[str, bool]


# Endpoints
@app.post("/transcribe")
async def transcribe_video(file: UploadFile = File(...)):
    try:
        file_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', Path(file.filename).stem)
        video_path = video_folder / f"{file_name}.mp4"
        audio_path = audio_folder / f"{file_name}.mp3"

        save_uploaded_file(file, video_path)
        convert_to_mp3(video_path, audio_path)

        transcription = whisper_api.transcribe(str(audio_path))
        qa_output = transcription_to_qa(transcription)

        os.remove(video_path)
        os.remove(audio_path)

        return {"id": file_name, "transcription": qa_output}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing video: {e}")

@app.post("/analyze-sentiment")
async def analyze_sentiment(data: SentimentRequest):
    try:
        human_responses = extract_human_responses(data.transcription)
        sentiment_result = emotion_analyzer.analyze_emotions(human_responses)
        response_id = data.id or str(uuid.uuid4())

        return {"id": response_id, "sentiment": sentiment_result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing sentiment: {e}")


@app.post("/check-human-words")
async def check_human_words(data: HumanWordCheckRequest, scenarios: List[str]):
    try:
       
        # Extract only the human responses
        human_responses = " ".join([qa.human.strip() for qa in data.transcription if qa.human.strip()])

        # Initialize the BagOfWordsFilter with the user-provided words
        word_filter = BagOfWordsFilter(data.words_to_check)

        # Check for each word's presence in the combined human responses
        words_found = word_filter.check_sentence(human_responses)
        # Format transcription as a string
        transcription_str = "\n".join(
            [f"Bot: {qa.bot}\nHuman: {qa.human}" for qa in data.transcription]
        )
        # Ensure all results are boolean and formatted correctly
        validated_words_found = {word: bool(words_found.get(word, False)) for word in data.words_to_check}
        
        # Prepare system prompt for OpenAI API
        system_prompt = (
            "You are a conversation analysis expert. Analyze the given conversation based on the following scenarios: "
            + ", ".join(scenarios) + ". do analyis human moods only based on questions by bot"
        )
        
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": transcription_str}
            ]
        )
        # Extract response from OpenAI
        conversation_analysis = response.choices[0].message.content
        # Combine results into a JSON response
        response_data = {
            "id": data.id,
            "words_found": validated_words_found,
            "conversation_analysis": conversation_analysis
        }

        return response_data

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"Error processing human word check and analysis: {e}")


@app.post("/update-sentiment")
async def update_sentiment(sentiments: Dict[str, str]):
    """
    Endpoint to define new sentiments and their definitions in JSON format. Older sentiments are cleared.
    """
    try:
        # Overwrite the sentiments file with the new sentiments
        with open(custom_sentiments_file, "w") as f:
            json.dump(sentiments, f, indent=4)

        return {
            "message": "Sentiment definitions updated successfully."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error defining sentiments: {e}")
    
    
@app.post("/analyze-sentiment-with-custom")
async def analyze_sentiment_with_custom(data: SentimentRequest):
    """
    Analyze sentiment using both predefined and user-defined sentiments.
    If a QAPair has no human answer, the sentiment is set to "none" without calling the API.
    """
    try:
        # Load user-defined sentiments
        with open(custom_sentiments_file, "r") as f:
            custom_sentiments = json.load(f)

        if not isinstance(data.transcription, list):
            raise ValueError("The 'transcription' field must be a list of conversations.")

        responses_with_sentiment = []

        for i, conversation in enumerate(data.transcription):
            # Convert the Pydantic model instance to a dict to access all fields.
            conv = conversation.dict()
            timestamp_start = conv.get("timestamp_start", "")
            timestamp_end = conv.get("timestamp_end", "")
            question = conv.get("bot", "").strip()
            answer = conv.get("human", "").strip()

            print(f"Extracted timestamps for entry {i}: start={timestamp_start}, end={timestamp_end}")

            if not question:
                raise ValueError(f"Bot question is empty in entry at index {i}. Entry: {conv}")

            if not answer:
                responses_with_sentiment.append({
                    "timestamp_start": timestamp_start,
                    "timestamp_end": timestamp_end,
                    "question": question,
                    "answer": "none",
                    "sentiment": "none"
                })
                continue

            system_prompt = (
                "You are a sentiment analysis expert. Analyze the sentiment of the given text using the following custom sentiments: " +
                ", ".join([f"'{name}': {definition}" for name, definition in custom_sentiments.items()]) +
                ". Only provide the most appropriate sentiment from the above mentioned options. Do not explain or write extra text—give only one sentiment name."
            )

            response = client.chat.completions.create(
                model="gpt-4-turbo",
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Question: {question}\nAnswer: {answer}"}
                ]
            )

            sentiment = response.choices[0].message.content.strip()

            responses_with_sentiment.append({
                "timestamp_start": timestamp_start,
                "timestamp_end": timestamp_end,
                "question": question,
                "answer": answer,
                "sentiment": sentiment
            })

        return {"id": data.id or str(uuid.uuid4()), "responses_with_sentiment": responses_with_sentiment}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing sentiment with custom definitions: {e}")

@app.post("/transcribe-media")
async def transcribe_media(file: UploadFile = File(...)):
    try:
        file_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', Path(file.filename).stem)
        file_extension = Path(file.filename).suffix.lower()
        file_path = video_folder / f"{file_name}{file_extension}"
        save_uploaded_file(file, file_path)

        # Convert if needed
        if file_extension in [".mp4", ".mov", ".avi"]:
            audio_path = audio_folder / f"{file_name}.mp3"
            convert_to_mp3(file_path, audio_path)
            transcription_result = process_transcription(audio_path, file_name)
            os.remove(audio_path)
        elif file_extension in [".mp3", ".wav"]:
            transcription_result = process_transcription(file_path, file_name)
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format. Please upload MP3, WAV, or MP4 files."
            )

        os.remove(file_path)

        # **If a "message" is present, you can decide how to return it.**
        if "message" in transcription_result:
            # e.g. just return the whole dict as is
            return transcription_result

        return transcription_result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing media: {e}")


@app.post("/process_video/")
async def process_video(
    video: UploadFile = File(...),
    start_time: float = Form(...),
    end_time: float = Form(...),
):
    original_video_path = f"temp_{video.filename}"
    cropped_video_by_time_path = f"cropped_time_{video.filename}"
    cropped_video_path = f"cropped_human_{video.filename}"

    with open(original_video_path, "wb") as f:
        f.write(await video.read())

    try:
        # Crop the video by time
        crop_video_by_time(original_video_path, cropped_video_by_time_path, start_time, end_time)

        # Extract human top-left region
        extract_human_top_left(cropped_video_by_time_path, cropped_video_path)

    except ValueError as e:
        os.remove(original_video_path)
        if os.path.exists(cropped_video_by_time_path):
            os.remove(cropped_video_by_time_path)
        return {"error": str(e)}

    # Load the FES model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = AutoFeatureExtractor.from_pretrained("dima806/facial_emotions_image_detection")
    model = AutoModelForImageClassification.from_pretrained("dima806/facial_emotions_image_detection").to(device)

    # Process the cropped video
    expressions = process_video_segment(cropped_video_path, model, feature_extractor, device)

    # Clean up temporary files
    os.remove(original_video_path)
    os.remove(cropped_video_by_time_path)
    os.remove(cropped_video_path)

    return {"expressions": expressions}
@app.post("/process-data/")
async def process_data(data: CombinedInput):
    try:
        # Extract relevant data
        sentiment_responses = data.sentiment_data.responses_with_sentiment
        facial_expression = data.facial_data.expressions

        # Use utility function to process data
        output = prioritize_facial_expression(facial_expression, sentiment_responses)

        return {"result": output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/process_video_1/")
async def process_video_1(
    video: UploadFile = File(...), 
    transcription: str = Form(...)
):
    original_video_path = f"temp_{video.filename}"
    cropped_video_by_time_path = f"cropped_time_{video.filename}"

    with open(original_video_path, "wb") as f:
        f.write(await video.read())

    try:
        # Parse the transcription field into a Python dictionary
        transcription_data = json.loads(transcription)

        if "transcription" not in transcription_data:
            raise ValueError("Invalid transcription format: 'transcription' key is missing.")

        results = []

        # Load the FES model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        feature_extractor = AutoFeatureExtractor.from_pretrained("dima806/facial_emotions_image_detection")
        model = AutoModelForImageClassification.from_pretrained("dima806/facial_emotions_image_detection").to(device)

        for segment in transcription_data["transcription"]:
            try:
                start_time = float(segment["timestamp_start"].replace(" s", ""))
                end_time = float(segment["timestamp_end"].replace(" s", ""))

                # Generate a unique path for the cropped video segment
                cropped_segment_path = f"cropped_{start_time}_{end_time}_{video.filename}"

                # Crop the video by time for the current segment
                crop_video_by_time(original_video_path, cropped_segment_path, start_time, end_time)

                # Extract human top-left region for the current segment
                extract_human_top_left(cropped_segment_path, cropped_video_by_time_path)

                # Process the cropped video and extract expressions
                expressions = process_video_segment(cropped_video_by_time_path, model, feature_extractor, device)

                # Append the results for this segment
                results.append({
                    "start_time": start_time,
                    "end_time": end_time,
                    "expressions": expressions
                })

                # Clean up temporary cropped segment files
                os.remove(cropped_segment_path)
                os.remove(cropped_video_by_time_path)

            except Exception as e:
                results.append({
                    "start_time": segment.get("timestamp_start"),
                    "end_time": segment.get("timestamp_end"),
                    "error": str(e)
                })

    finally:
        # Clean up the original video file
        os.remove(original_video_path)

    return {"results": results}
@app.post("/process-data_final/")
async def process_data_final(request: Request):
    try:
        # Parse the incoming JSON
        body = await request.json()

        # Validate required keys
        if "sentiment_data" not in body or "facial_data" not in body:
            raise HTTPException(status_code=400, detail="Missing 'sentiment_data' or 'facial_data'.")

        sentiment_data = body["sentiment_data"]
        facial_data = body["facial_data"]

        if "responses_with_sentiment" not in sentiment_data or "expressions" not in facial_data:
            raise HTTPException(
                status_code=400,
                detail="Missing 'responses_with_sentiment' or 'expressions' in input."
            )

        # Extract data
        sentiment_responses = sentiment_data["responses_with_sentiment"]
        facial_expressions = facial_data["expressions"]

        # Validate lengths
        if len(sentiment_responses) != len(facial_expressions):
            raise HTTPException(
                status_code=400,
                detail="Mismatch in the number of sentiment responses and facial expressions."
            )

        # Prepare detailed results
        results = []

        for sentiment, fer in zip(sentiment_responses, facial_expressions):
            # Ensure the FER and sentiment are in the correct format
            facial_expression = fer.get("expressions", "unknown") if "error" not in fer else fer["error"]
            result = prioritize_facial_expression(facial_expression, [sentiment])
            results.append(result)

        return {"results": results}

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON input.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-video-data")
async def process_video_data(file: UploadFile = File(...)):
    try:
        # Step 1: Transcribe the video
        transcribe_response = await transcribe_media(file)

        # If the transcriber returned a message indicating missing/empty audio, exit early.
        if "message" in transcribe_response:
            return {
                "id": transcribe_response.get("id", "unknown_id"),
                "message": transcribe_response["message"],
                "transcription": []
            }
        if "id" not in transcribe_response or "transcription" not in transcribe_response:
            return {
                "id": "unknown_id",
                "message": "Transcription response is invalid.",
                "transcription": []
            }

        # ----------------------------
        # Step 2: Analyze sentiment with custom definitions
        # ----------------------------
        transcription = transcribe_response["transcription"]
        transcription_id = transcribe_response["id"]

        sentiment_request = SentimentRequest(id=transcription_id, transcription=transcription)
        sentiment_response = await analyze_sentiment_with_custom(sentiment_request)
        if "responses_with_sentiment" not in sentiment_response:
            return {
                "id": transcription_id,
                "message": "Sentiment response is invalid.",
                "transcription": []
            }
        responses_with_sentiment = sentiment_response["responses_with_sentiment"]

        # ----------------------------
        # Step 3: Process the video for facial expressions for ALL segments
        # ----------------------------
        try:
            results = []
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            feature_extractor = AutoFeatureExtractor.from_pretrained("dima806/facial_emotions_image_detection")
            model = AutoModelForImageClassification.from_pretrained("dima806/facial_emotions_image_detection").to(device)

            # Create a temporary file for the uploaded video content
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
                await file.seek(0)
                shutil.copyfileobj(file.file, temp_file)
                temp_file_path = temp_file.name

            # Process each transcription segment regardless of sentiment
            for i, segment in enumerate(transcription):
                try:
                    # Extract timestamps - handle both string and float formats
                    if isinstance(segment["timestamp_start"], str) and "s" in segment["timestamp_start"]:
                        start_time = float(segment["timestamp_start"].replace(" s", ""))
                    else:
                        start_time = float(segment["timestamp_start"])
                        
                    if isinstance(segment["timestamp_end"], str) and "s" in segment["timestamp_end"]:
                        end_time = float(segment["timestamp_end"].replace(" s", ""))
                    else:
                        end_time = float(segment["timestamp_end"])
                    
                    # Skip if the timestamps are invalid
                    if start_time >= end_time or end_time <= 0:
                        results.append({
                            "start_time": segment.get("timestamp_start"),
                            "end_time": segment.get("timestamp_end"),
                            "error": "Invalid timestamps",
                            "expressions": ""
                        })
                        continue
                    
                    # Create temporary files for cropped segments
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_segment, \
                         tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_cropped:
                        cropped_segment_path = temp_segment.name
                        cropped_video_by_time_path = temp_cropped.name

                    # Crop the video by time and extract the human region
                    crop_video_by_time(temp_file_path, cropped_segment_path, start_time, end_time)
                    
                    try:
                        extract_human_top_left(cropped_segment_path, cropped_video_by_time_path)
                        expressions = process_video_segment(cropped_video_by_time_path, model, feature_extractor, device)
                    except ValueError as ve:
                        # If face detection fails, still record the error but continue processing
                        results.append({
                            "start_time": start_time,
                            "end_time": end_time,
                            "error": str(ve),
                            "expressions": ""
                        })
                        os.unlink(cropped_segment_path)
                        if os.path.exists(cropped_video_by_time_path):
                            os.unlink(cropped_video_by_time_path)
                        continue
                    
                    results.append({
                        "start_time": start_time,
                        "end_time": end_time,
                        "expressions": expressions,
                    })

                    os.unlink(cropped_segment_path)
                    os.unlink(cropped_video_by_time_path)
                    
                except Exception as e:
                    results.append({
                        "start_time": segment.get("timestamp_start"),
                        "end_time": segment.get("timestamp_end"),
                        "error": str(e),
                        "expressions": ""
                    })

            # Clean up the temporary original file
            os.unlink(temp_file_path)

        except Exception as e:
            return {
                "id": transcription_id,
                "message": f"Error processing video segments: {str(e)}",
                "transcription": []
            }

        # ----------------------------
        # Step 4: Process sentiment and facial data together
        # ----------------------------
        final_results = []
        overall_sentiment_data = []
        overall_score_total = 0
        
        # Make sure we have matching lengths or handle mismatches
        if len(responses_with_sentiment) != len(results):
            # Handle mismatched lengths by using what we have
            max_len = min(len(responses_with_sentiment), len(results))
            responses_with_sentiment = responses_with_sentiment[:max_len]
            results = results[:max_len]
        
        # Process each pair of sentiment and facial expression data
        for i, (sentiment, fer) in enumerate(zip(responses_with_sentiment, results)):
            # Extract facial expression or error message
            facial_expression = fer.get("expressions", "")
            if "error" in fer:
                facial_expression = fer["error"]
            
            # Process the current pair
            result = prioritize_facial_expression(facial_expression, [sentiment])
            final_results.append(result)
            
            # Collect data for overall sentiment
            if result["accumulative_sentiment"] != "none":
                overall_sentiment_data.append(result["accumulative_sentiment"])
                overall_score_total += result["impression_score"]
        
        # Calculate overall sentiment for the entire conversation
        overall_sentiment = "neutral"
        overall_impression_score = 50
        
        if overall_sentiment_data:
            try:
                # Get overall sentiment from OpenAI
                overall_prompt = (
                    f"Analyze these individual sentiment descriptions: {', '.join(overall_sentiment_data)}. "
                    "Provide an overall 1-2 word sentiment that best summarizes all of them. "
                    "Consider the frequency and intensity of each sentiment. "
                    "Provide only the final 1-2 word description with no additional text or explanation."
                )
                
                overall_response = client.chat.completions.create(
                    model="gpt-4-turbo",
                    temperature=0,
                    messages=[{"role": "system", "content": overall_prompt}]
                )
                
                # Update the overall description
                overall_sentiment = overall_response.choices[0].message.content.strip()
                overall_impression_score = min(overall_score_total // max(1, len(overall_sentiment_data)), 100)
                
            except Exception as e:
                print(f"Error getting final overall sentiment: {str(e)}")
                # Use default values if error occurs

        # ----------------------------
        # Step 5: Return comprehensive results
        # ----------------------------
        return {
            "id": transcription_id,
            "transcription": transcription,
            "sentiment_analysis": responses_with_sentiment,
            "facial_expressions": results,
            "detailed_results": final_results,
            "overall_analysis": {
                "accumulative_sentiment": overall_sentiment,
                "impression_score": overall_impression_score
            }
        }

    except Exception as e:
        return {
            "id": "unknown_id",
            "message": f"Error processing video data: {str(e)}",
            "transcription": []
        }