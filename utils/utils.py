import os
import cv2
import mediapipe as mp
import torch
from facenet_pytorch import MTCNN
import time 
from fastapi import File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
from moviepy.editor import VideoFileClip
from pathlib import Path
import re
import subprocess
import assemblyai as aai
import openai
from openai import OpenAI
openai.api_key =os.getenv("OPENAI_API_KEY1")
def crop_video_by_time(video_path, output_path, start_time, end_time):
    """
    Crop a video based on start and end time using FFmpeg.
    """
    duration = end_time - start_time
    ffmpeg_command = (
        f'ffmpeg -i "{video_path}" -ss {start_time} -t {duration} -c:v libx264 -c:a aac "{output_path}" -y'
    )
    os.system(ffmpeg_command)

def extract_human_top_left(video_path, output_path, margin=0.05, frame_tolerance=40):
    """
    Detect and crop the human segment (top-left corner) of a video using a more accurate face detector (MTCNN).
    If no face is detected in a frame, it will check the next few frames in the time slot (up to frame_tolerance).
    
    Args:
        video_path (str): Path to the input video.
        output_path (str): Path to save the cropped video.
        margin (float): Extra margin to add around the detected face bounding box as a fraction.
        frame_tolerance (int): Number of consecutive frames to check for a face before deciding no face is present.
        
    Raises:
        ValueError: If no face is detected in the ROI of the video.
    """
    # Determine processing device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Initialize the MTCNN face detector
    mtcnn = MTCNN(keep_all=True, device=device)
    
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise ValueError("Unable to open video file.")
    
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the region-of-interest (top-left corner) as 35% of the frame width and height.
    roi_x_end = int(frame_width * 0.35)
    roi_y_end = int(frame_height * 0.35)

    human_crops = []
    frame_skipped = 0  # To track the number of frames where no face is detected in the current time slot

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        # Crop region-of-interest from the frame
        roi = frame[0:roi_y_end, 0:roi_x_end]
        # Convert the ROI from BGR to RGB as required by MTCNN
        rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        
        # Detect faces in the ROI
        boxes, probs = mtcnn.detect(rgb_roi)
        
        # If faces are detected, process the first detected box (or all boxes as needed)
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = box
                # Expand the bounding box by the specified margin
                w = x2 - x1
                h = y2 - y1
                new_x1 = max(0, int(x1 - margin * w))
                new_y1 = max(0, int(y1 - margin * h))
                new_x2 = min(roi_x_end, int(x2 + margin * w))
                new_y2 = min(roi_y_end, int(y2 + margin * h))
                human_crops.append((new_x1, new_y1, new_x2, new_y2))
                # If you only want the first detection per frame, uncomment the next line.
                break
            frame_skipped = 0  # Reset the frame skipped count after detecting a face
        else:
            frame_skipped += 1  # Increment the count for frames where no face is detected
        
        # If we've skipped too many frames, stop looking further in this time window
        if frame_skipped >= frame_tolerance:
            print(f"Skipped {frame_skipped} frames without detecting a face.")
            frame_skipped = 0  # Reset the counter for the next time slot

    video.release()

    if not human_crops:
        raise ValueError("No faces detected in the specified ROI of the video.")

    # Aggregate detections by averaging over collected bounding boxes
    x_min = int(sum(crop[0] for crop in human_crops) / len(human_crops))
    y_min = int(sum(crop[1] for crop in human_crops) / len(human_crops))
    x_max = int(sum(crop[2] for crop in human_crops) / len(human_crops))
    y_max = int(sum(crop[3] for crop in human_crops) / len(human_crops))

    crop_width = x_max - x_min
    crop_height = y_max - y_min

    # Build FFmpeg crop filter command
    crop_filter = f"crop={crop_width}:{crop_height}:{x_min}:{y_min}"
    ffmpeg_command = f'ffmpeg -i "{video_path}" -filter:v "{crop_filter}" -c:a copy "{output_path}" -y'
    os.system(ffmpeg_command)
def process_video_segment(video_path, model, feature_extractor, device):
    """
    Process the entire video and return a single FES label for the entire video.
    """
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    aggregated_logits = None  # To aggregate logits for the entire video
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = feature_extractor(images=[rgb_frame], return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits.squeeze(0)  # Get logits for the frame

        # Aggregate logits across all frames
        if aggregated_logits is None:
            aggregated_logits = logits
        else:
            aggregated_logits += logits

        frame_count += 1

    cap.release()

    if frame_count == 0:
        raise ValueError("No frames processed from the video.")

    # Compute the final predicted class from aggregated logits
    aggregated_logits /= frame_count  # Optionally normalize by the number of frames
    predicted_class = torch.argmax(aggregated_logits).item()
    final_label = model.config.id2label[predicted_class]

    return final_label


# Pydantic Models
class QAPair(BaseModel):
    bot: Optional[str]
    human: Optional[str]


# Helper Functions
def save_uploaded_file(file: UploadFile, destination: Path):
    """Save uploaded file to the specified destination."""
    with open(destination, "wb") as f:
        f.write(file.file.read())

def extract_human_responses(transcription: List[QAPair]) -> str:
    """Extract and concatenate human responses from transcription."""
    return " ".join(qa.human for qa in transcription if qa.human)
def preprocess_video(video_path: Path, output_path: Path):
    """Preprocess video to ensure compatibility """
    try:
        cmd = [
            "ffmpeg", "-y", "-i", str(video_path), 
            "-c:v", "libx264", "-preset", "fast", "-c:a", "aac", "-strict", "experimental",
            str(output_path)
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Error preprocessing video: {e.stderr.decode()}")

def convert_to_mp3(video_path: Path, audio_path: Path):
    """Convert a video file to MP3 format."""
    try:
        temp_video = video_path.parent / f"temp_{video_path.name}"
        preprocess_video(video_path, temp_video)

        clip = VideoFileClip(str(temp_video))
        if not clip.audio:
            raise HTTPException(status_code=500, detail="No audio stream found in the video")
        clip.audio.write_audiofile(str(audio_path), codec='libmp3lame')
        clip.close()

        temp_video.unlink()  # Delete temporary processed video
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error converting video to MP3: {e}")
def process_transcription(file_path: Path, file_id: str, speakers_expected=2, retries=3, retry_delay=2):
    try:
        # Transcription configuration with speaker_labels enabled
        config = aai.TranscriptionConfig(speaker_labels=True, speech_model=aai.SpeechModel.slam_1)
        transcriber = aai.Transcriber()

        # Attempt to transcribe the file with retry logic
        for attempt in range(retries):
            try:
                # Perform transcription
                transcript = transcriber.transcribe(str(file_path), config)
                print(f"Transcription successful on attempt {attempt + 1}")
                
                # If no utterances found, return a missing/empty audio message
                if not transcript.utterances:
                    return {
                        "id": file_id,
                        "message": "No audio or empty audio found in the file."
                    }

                # Debug print: Show all utterances from AssemblyAI with their speaker labels
                print("\n--- RAW SPEAKER DIARIZATION FROM ASSEMBLY AI ---")
                all_speakers = set()
                for utterance in transcript.utterances:
                    speaker = utterance.speaker.lower()
                    all_speakers.add(speaker)
                    print(f"Speaker {speaker.upper()}: {utterance.text.strip()} [{utterance.start/1000:.2f}s - {utterance.end/1000:.2f}s]")
                print(f"Total speakers detected: {len(all_speakers)} - {', '.join(s.upper() for s in all_speakers)}")
                print("---------------------------------------------\n")

                # Special case: If only one speaker is detected, we need to analyze the content
                # to determine if it's a bot or human based on short responses
                if len(all_speakers) == 1:
                    print("Special case: Only one speaker detected")
                    
                    # Analyze the content to see if we can split it into bot and human parts
                    # Look for patterns like "Question? Answer." where answers are typically short
                    single_speaker = list(all_speakers)[0]
                    all_text = " ".join([u.text.strip() for u in transcript.utterances])
                    
                    # Try to find a pattern of questions followed by short answers (Yes, No, etc.)
                    qa_pattern = re.compile(r'([^.!?]+[.!?])\s+(Yes|No|Correct|Right|Sure|Okay|OK|I agree|Agreed|I do)[\s\.]', re.IGNORECASE)
                    matches = list(qa_pattern.finditer(all_text))
                    
                    if matches:
                        print(f"Found {len(matches)} potential question-answer pairs in single speaker text")
                        # We can split this into bot and human parts
                        conversation_data = {
                            "id": file_id,
                            "transcription": []
                        }
                        
                        for match in matches:
                            bot_text = match.group(1).strip()
                            human_text = match.group(2).strip()
                            
                            # Add to conversation data
                            conversation_data["transcription"].append({
                                "timestamp_start": "0.00 s",  # We don't have precise timestamps for these splits
                                "timestamp_end": "0.00 s",
                                "speaker": "bot",
                                "text": bot_text,
                                "bot": bot_text,
                                "human": human_text
                            })
                        
                        print("\n--- SPLIT SINGLE SPEAKER INTO Q&A PAIRS ---")
                        for i, entry in enumerate(conversation_data["transcription"]):
                            print(f"Entry {i+1}:")
                            print(f"  Bot: {entry['bot']}")
                            print(f"  Human: {entry['human']}")
                        print("----------------------------------\n")
                        
                        return conversation_data
                    else:
                        # If we can't detect Q&A pairs, assume it's all bot content
                        print("No clear Q&A pairs found, treating entire content as bot")
                        conversation_data = {
                            "id": file_id,
                            "transcription": []
                        }
                        
                        # Split by sentences to create multiple entries if needed
                        sentences = re.split(r'(?<=[.!?])\s+', all_text)
                        
                        # Group sentences into reasonable bot statements (avoid too many tiny entries)
                        grouped_sentences = []
                        current_group = ""
                        
                        for sentence in sentences:
                            if len(current_group) + len(sentence) < 200:  # Keep groups reasonably sized
                                current_group += " " + sentence if current_group else sentence
                            else:
                                if current_group:
                                    grouped_sentences.append(current_group)
                                current_group = sentence
                        
                        if current_group:  # Add the last group
                            grouped_sentences.append(current_group)
                        
                        # If still no groups, use the whole text as one entry
                        if not grouped_sentences:
                            grouped_sentences = [all_text]
                        
                        # Create conversation entries
                        for i, group in enumerate(grouped_sentences):
                            conversation_data["transcription"].append({
                                "timestamp_start": f"{i * 10}.00 s",  # Approximate timestamps
                                "timestamp_end": f"{(i + 1) * 10}.00 s",
                                "speaker": "bot",
                                "text": group.strip(),
                                "bot": group.strip(),
                                "human": ""  # Empty human response
                            })
                        
                        print("\n--- TREATED SINGLE SPEAKER AS BOT ---")
                        for i, entry in enumerate(conversation_data["transcription"]):
                            print(f"Entry {i+1}:")
                            print(f"  Bot: {entry['bot']}")
                            print(f"  Human: {entry['human']}")
                        print("----------------------------------\n")
                        
                        return conversation_data
                
                # Normal case with multiple speakers:
                # Calculate statistics for each speaker, focusing on utterance lengths
                speaker_stats = {}
                for utterance in transcript.utterances:
                    speaker = utterance.speaker.lower()
                    text = utterance.text.strip()
                    words = text.split()
                    word_count = len(words)
                    
                    if speaker not in speaker_stats:
                        speaker_stats[speaker] = {
                            "utterances": [],
                            "short_utterance_count": 0,  # Count of utterances with 1-3 words
                            "total_utterance_count": 0,
                            "short_utterance_ratio": 0.0
                        }
                    
                    speaker_stats[speaker]["utterances"].append({
                        "text": text,
                        "word_count": word_count,
                        "is_short": word_count <= 3
                    })
                    
                    speaker_stats[speaker]["total_utterance_count"] += 1
                    if word_count <= 3:
                        speaker_stats[speaker]["short_utterance_count"] += 1
                
                # Calculate short utterance ratio for each speaker
                for speaker in speaker_stats:
                    if speaker_stats[speaker]["total_utterance_count"] > 0:
                        speaker_stats[speaker]["short_utterance_ratio"] = (
                            speaker_stats[speaker]["short_utterance_count"] / 
                            speaker_stats[speaker]["total_utterance_count"]
                        )
                
                # Print speaker statistics
                print("\n--- SPEAKER STATISTICS ---")
                for speaker in speaker_stats:
                    print(f"Speaker {speaker.upper()}:")
                    print(f"  - Total utterances: {speaker_stats[speaker]['total_utterance_count']}")
                    print(f"  - Short utterances (1-3 words): {speaker_stats[speaker]['short_utterance_count']}")
                    print(f"  - Short utterance ratio: {speaker_stats[speaker]['short_utterance_ratio']:.2f}")
                    
                    # Print some examples of utterances
                    print("  - Sample utterances:")
                    for i, utt in enumerate(speaker_stats[speaker]["utterances"][:5]):  # Print up to 5 examples
                        print(f"      {i+1}. \"{utt['text']}\" ({utt['word_count']} words)")
                    if len(speaker_stats[speaker]["utterances"]) > 5:
                        print(f"      ... and {len(speaker_stats[speaker]['utterances']) - 5} more")
                print("------------------------\n")
                
                # Determine the human speaker - speaker with highest ratio of short utterances
                # Only consider speakers with at least 2 utterances to avoid flukes
                valid_speakers = {s: stats for s, stats in speaker_stats.items() 
                                if stats["total_utterance_count"] >= 2}
                
                if not valid_speakers:
                    # Fall back to all speakers if none have at least 2 utterances
                    valid_speakers = speaker_stats
                
                # Find the speaker with the highest ratio of short utterances
                human_speaker = max(valid_speakers, 
                                   key=lambda s: valid_speakers[s]["short_utterance_ratio"],
                                   default=None)
                
                # Only consider as human if at least 30% of utterances are short OR
                # if it has at least 2 short utterances
                is_valid_human = False
                if human_speaker:
                    if (valid_speakers[human_speaker]["short_utterance_ratio"] >= 0.3 or
                        valid_speakers[human_speaker]["short_utterance_count"] >= 2):
                        is_valid_human = True
                
                # All other speakers are bots
                if is_valid_human:
                    bot_speakers = [s for s in speaker_stats if s != human_speaker]
                else:
                    # If no valid human found, pick the speaker with the lowest utterance length
                    # as human and rest as bots
                    human_speaker = min(valid_speakers, 
                                      key=lambda s: sum(u["word_count"] for u in valid_speakers[s]["utterances"]) / 
                                                   valid_speakers[s]["total_utterance_count"],
                                      default=None)
                    bot_speakers = [s for s in speaker_stats if s != human_speaker]
                
                print(f"Identified human speaker: {human_speaker.upper() if human_speaker else 'None'}")
                print(f"Identified bot speaker(s): {', '.join(s.upper() for s in bot_speakers)}")
                
                # Create a mapping of original speakers to roles
                speaker_roles = {s: "bot" for s in bot_speakers}
                if human_speaker:
                    speaker_roles[human_speaker] = "human"
                
                # Create the structured conversation data
                conversation_data = {
                    "id": file_id,
                    "transcription": []
                }
                
                # First pass: Process utterances with correct speaker labels
                processed_utterances = []
                for utterance in transcript.utterances:
                    speaker = utterance.speaker.lower()
                    role = speaker_roles.get(speaker, "bot")  # Default to bot if unknown
                    
                    processed_utterances.append({
                        "timestamp_start": f"{utterance.start / 1000:.2f} s",
                        "timestamp_end": f"{utterance.end / 1000:.2f} s",
                        "speaker": role,
                        "text": utterance.text.strip(),
                        "original_speaker": speaker
                    })
                
                # Debug: Print processed utterances with assigned roles
                print("\n--- INITIAL SPEAKER ROLE ASSIGNMENT ---")
                for i, utt in enumerate(processed_utterances):
                    print(f"{i+1}. [{utt['original_speaker'].upper()} → {utt['speaker']}]: {utt['text']}")
                print("------------------------------------\n")
                
                # Second pass: Combine consecutive human utterances
                i = 0
                combined_utterances = []
                while i < len(processed_utterances):
                    current = processed_utterances[i]
                    
                    # If current utterance is human, check for consecutive human utterances
                    if current["speaker"] == "human":
                        combined_text = current["text"]
                        start_time = current["timestamp_start"]
                        end_time = current["timestamp_end"]
                        original_speakers = [current["original_speaker"]]
                        
                        # Look ahead for consecutive human utterances
                        j = i + 1
                        while j < len(processed_utterances) and processed_utterances[j]["speaker"] == "human":
                            combined_text += " " + processed_utterances[j]["text"]
                            end_time = processed_utterances[j]["timestamp_end"]
                            original_speakers.append(processed_utterances[j]["original_speaker"])
                            j += 1
                        
                        # Create a combined human utterance
                        combined_utterances.append({
                            "timestamp_start": start_time,
                            "timestamp_end": end_time,
                            "speaker": "human",
                            "text": combined_text,
                            "original_speakers": original_speakers
                        })
                        
                        i = j  # Skip the utterances we just combined
                    else:
                        # For bot utterances, add them as-is
                        combined_utterances.append({
                            "timestamp_start": current["timestamp_start"],
                            "timestamp_end": current["timestamp_end"],
                            "speaker": "bot",
                            "text": current["text"],
                            "original_speakers": [current["original_speaker"]]
                        })
                        i += 1
                
                # Debug: Print combined utterances
                print("\n--- AFTER COMBINING CONSECUTIVE HUMAN UTTERANCES ---")
                for i, utt in enumerate(combined_utterances):
                    print(f"{i+1}. [{'/'.join(s.upper() for s in utt['original_speakers'])} → {utt['speaker']}]: {utt['text']}")
                print("------------------------------------------------\n")
                
                # Third pass: Create bot-human pairs or handle missing responses
                i = 0
                bot_human_pairs = []
                while i < len(combined_utterances):
                    current = combined_utterances[i]
                    
                    if current["speaker"] == "bot":
                        # Look for a human response
                        human_response = ""
                        human_timestamp_end = current["timestamp_end"]
                        
                        if i + 1 < len(combined_utterances) and combined_utterances[i + 1]["speaker"] == "human":
                            human_response = combined_utterances[i + 1]["text"]
                            human_timestamp_end = combined_utterances[i + 1]["timestamp_end"]
                            i += 2  # Skip both bot and human
                        else:
                            # No human response found
                            i += 1  # Skip just the bot
                        
                        # Add the pair to the conversation data
                        bot_human_pairs.append({
                            "timestamp_start": current["timestamp_start"],
                            "timestamp_end": human_timestamp_end,
                            "speaker": "bot",
                            "text": current["text"],
                            "bot": current["text"],
                            "human": human_response,
                            "original_speakers": current["original_speakers"]
                        })
                    elif current["speaker"] == "human" and (i == 0 or combined_utterances[i-1]["speaker"] != "bot"):
                        # Human utterance without a preceding bot utterance
                        bot_human_pairs.append({
                            "timestamp_start": current["timestamp_start"],
                            "timestamp_end": current["timestamp_end"],
                            "speaker": "human",
                            "text": current["text"],
                            "bot": "",  # Empty bot response 
                            "human": current["text"],
                            "original_speakers": current["original_speakers"]
                        })
                        i += 1
                    else:
                        # This should not happen with proper processing, but just in case
                        i += 1
                
                # Special case: If human starts without bot (leading human utterance),
                # flip it so bot is not empty
                if bot_human_pairs and "bot" in bot_human_pairs[0] and bot_human_pairs[0]["bot"] == "":
                    print("Converting leading human utterance to bot to avoid empty bot field")
                    bot_human_pairs[0]["bot"] = bot_human_pairs[0]["human"]
                    bot_human_pairs[0]["human"] = ""
                    bot_human_pairs[0]["speaker"] = "bot"
                
                # Add all valid pairs to the final conversation data
                for pair in bot_human_pairs:
                    if "original_speakers" in pair:
                        del pair["original_speakers"]  # Remove debug field
                    conversation_data["transcription"].append(pair)
                
                # Debug: Print final conversation structure
                print("\n--- FINAL CONVERSATION STRUCTURE ---")
                for i, entry in enumerate(conversation_data["transcription"]):
                    print(f"Entry {i+1}:")
                    print(f"  Bot: {entry['bot']}")
                    print(f"  Human: {entry['human']}")
                print("----------------------------------\n")
                
                # If no valid entries were created but there is transcription text,
                # add at least one entry to ensure data is not lost
                if not conversation_data["transcription"] and transcript.utterances:
                    first_utt = transcript.utterances[0]
                    # IMPORTANT: For fallback, always make it a bot utterance to avoid empty bot field
                    conversation_data["transcription"].append({
                        "timestamp_start": f"{first_utt.start / 1000:.2f} s",
                        "timestamp_end": f"{first_utt.end / 1000:.2f} s",
                        "speaker": "bot",
                        "text": first_utt.text.strip(),
                        "bot": first_utt.text.strip(),
                        "human": ""  # Empty human response is okay
                    })
                    print("Added fallback entry to prevent data loss")

                return conversation_data

            except Exception as e:
                # If transcription fails, log the error and retry
                print(f"Attempt {attempt + 1} failed with error: {e}")
                if attempt < retries - 1:
                    time.sleep(retry_delay)  # Wait before retrying
                else:
                    raise HTTPException(status_code=500, detail=f"Error transcribing audio after {retries} attempts: {e}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing transcription: {e}")
def transcription_to_qa(transcription: str) -> List[Dict[str, str]]:
    """Convert raw transcription into a Q&A format."""
    try:
        segments = re.split(r'(?<=[?.])\s+', transcription.strip())
        qa_pairs = []
        current_question = None

        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue
            if segment.endswith("?"):
                if current_question:
                    qa_pairs.append({"bot": current_question, "human": ""})
                current_question = segment
            elif len(segment.split()) > 10:
                if current_question:
                    qa_pairs.append({"bot": current_question, "human": ""})
                qa_pairs.append({"bot": segment, "human": ""})
                current_question = None
            else:
                if current_question:
                    qa_pairs.append({"bot": current_question, "human": segment})
                    current_question = None
                else:
                    qa_pairs.append({"bot": segment, "human": ""})
        if current_question:
            qa_pairs.append({"bot": current_question, "human": ""})
        return qa_pairs

    except Exception as e:
        # Log or handle the exception as needed
        raise ValueError(f"An error occurred while processing the transcription: {e}")
    
def prioritize_facial_expression(facial_expression, sentiment_responses):
    """
    Analyze facial expression and sentiment data to determine the priority and accumulative sentiment.
    
    Args:
        facial_expression (str): The detected facial expression or error message
        sentiment_responses (list): List of sentiment analysis results
        
    Returns:
        dict: A dictionary containing the priority, facial expression, sentiment analysis, 
              accumulative sentiment, and impression score
    """
    # Initialize default values
    description = "neutral"  # Default description
    total_score = 0
    detailed_results = []
    
    # Check if facial_expression contains an error message
    is_fer_valid = isinstance(facial_expression, str) and "error" not in facial_expression.lower()
    
    # Check if we have sentiment data
    has_sentiment_data = any(sr.get("sentiment", "").strip().lower() != "none" for sr in sentiment_responses)
    
    # If no valid FER and no sentiment data, return "none" for everything
    if not is_fer_valid and not has_sentiment_data:
        return {
            "priority": "none",
            "facial_expression": facial_expression if isinstance(facial_expression, str) else "unknown",
            "sentiment_analysis": [],
            "accumulative_sentiment": "none",
            "impression_score": 0
        }
    
    # Process each sentiment response
    for sentiment_response in sentiment_responses:
        sentiment = sentiment_response.get("sentiment", "neutral").strip()
        question = sentiment_response.get("question", "")
        answer = sentiment_response.get("answer", "")
        
        # Skip processing if sentiment is "none" and FER is not valid
        if sentiment.lower() == "none" and not is_fer_valid:
            detailed_results.append({
                "question": question,
                "answer": answer,
                "sentiment": sentiment,
                "accumulative_sentiment": "none",
                "score": 0
            })
            continue
        
        try:
            # Prepare prompt based on what data we have
            if is_fer_valid and sentiment.lower() != "none":
                prompt = (
                    f"Analyze the given facial expression '{facial_expression}' and the sentiment '{sentiment}', "
                    "and determine the cumulative emotional impact and matching level. Combine the expression and sentiment to generate a description of the overall emotion, such as "
                    "'Neutral,' 'happy,' 'sad,' ' fearful,' etc.  only 1 word sentiment "
                    "Determine the impression score and cumulative sentiment based on the following rules:"
                    "\n- If the sentiments are positive or strongly happy, assign an impression score of 90."
                    "\n- If the sentiments are neutral, assign an impression score of exactly 50."
                    "\n- If the sentiments are somewhat happy or positive, assign an impression score in the range of 50 to 90."
                    "\n- If the sentiments are somewhat sad, negative, or fearful, assign an impression score in the range of 10 to 49."
                    "\n- If the sentiments are strongly negative, sad, or fearful, assign an impression score of 10."
                    "Combine the expression and sentiment to generate a description of the overall emotion, using a concise 1 or 2-word cumulative sentiment "
                    "(e.g., 'happy,' 'sad,' 'fearful','angry'). "
                    "Provide the output in the following format exactly:"
                    "\n1. [Impression Score (numeric)]"
                    "\n2. [Cumulative Sentiment (1  word)]"
                )
            elif is_fer_valid:
                # Only facial expression is valid
                prompt = (
                    f"Analyze the given facial expression '{facial_expression}' without any sentiment analysis. "
                    "Determine the impression score and emotion description based on the facial expression alone:"
                    "\n- If the expression is happy or positive, assign an impression score of 80-90."
                    "\n- If the expression is neutral, assign an impression score of exactly 50."
                    "\n- If the expression is somewhat positive, assign an impression score in the range of 50 to 80."
                    "\n- If the expression is somewhat negative, assign an impression score in the range of 20 to 49."
                    "\n- If the expression is strongly negative, assign an impression score of 10-20."
                    "Generate a concise 1 or 2-word description of the emotion (e.g., 'happy,' ' sad,' 'fearful'). "
                    "Provide the output in the following format exactly:"
                    "\n1. [Impression Score (numeric)]"
                    "\n2. [Cumulative Sentiment (1  word)]"
                )
            elif sentiment.lower() != "none":
                # Only sentiment is valid
                prompt = (
                    f"Analyze the given sentiment '{sentiment}' without any facial expression data. "
                    "Determine the impression score and emotion description based on the sentiment alone:"
                    "\n- If the sentiment is positive or strongly happy, assign an impression score of 80-90."
                    "\n- If the sentiment is neutral, assign an impression score of exactly 50."
                    "\n- If the sentiment is somewhat positive, assign an impression score in the range of 50 to 80."
                    "\n- If the sentiment is somewhat negative, assign an impression score in the range of 20 to 49."
                    "\n- If the sentiment is strongly negative, assign an impression score of 10-20."
                    "Generate a concise 1 or 2-word description of the emotion (e.g., 'happy,'  sad,' 'fearful','angry'). "
                    "Provide the output in the following format exactly:"
                    "\n1. [Impression Score (numeric)]"
                    "\n2. [Cumulative Sentiment (1 word)]"
                )
            else:
                # Neither is valid, skip OpenAI call
                detailed_results.append({
                    "question": question,
                    "answer": answer,
                    "sentiment": sentiment,
                    "accumulative_sentiment": "none",
                    "score": 0
                })
                continue
                
            # Call OpenAI API
            api_response = openai.chat.completions.create(
                model="gpt-4-turbo",
                temperature=0,
                messages=[{"role": "system", "content": prompt}]
            )
            result = api_response.choices[0].message.content.strip()
            print("API Result Content:", result)
            
            # Extract score and description
            score_match = re.search(r"1\.\s*(\d+)", result)
            description_match = re.search(r"2\.\s*(.*)", result)
            score_value = int(score_match.group(1)) if score_match else 0
            current_description = description_match.group(1).strip() if description_match else "neutral"
            
            # Update our description variable
            description = current_description
            
            # Add to total score
            total_score += score_value
            
            # Add to detailed results
            detailed_results.append({
                "question": question,
                "answer": answer,
                "sentiment": sentiment,
                "accumulative_sentiment": current_description,
                "score": score_value
            })
            
        except Exception as e:
            print(f"Error in OpenAI API call: {str(e)}")
            # Add entry with default values on error
            detailed_results.append({
                "question": question,
                "answer": answer,
                "sentiment": sentiment,
                "accumulative_sentiment": "neutral",  # Default on error
                "score": 50  # Default neutral score
            })
            total_score += 50  # Add default score to total
    
    # Calculate overall impression score
    impression_score = min(total_score // max(1, len(detailed_results)), 100)
    
    # If we have multiple entries, get an overall accumulative sentiment
    if len(detailed_results) > 1:
        try:
            # Get all individual accumulative sentiments
            all_sentiments = [entry["accumulative_sentiment"] for entry in detailed_results 
                             if entry["accumulative_sentiment"] != "none"]
            
            if all_sentiments:
                # Get overall sentiment from OpenAI
                overall_prompt = (
                    f"Analyze these individual sentiment descriptions: {', '.join(all_sentiments)}. "
                    "Provide an overall 1-2 word sentiment that best summarizes all of them. "
                    "Consider the frequency and intensity of each sentiment. "
                    "Provide only the final 1-2 word description with no additional text or explanation."
                )
                
                overall_response = openai.chat.completions.create(
                    model="gpt-4-turbo",
                    temperature=0,
                    messages=[{"role": "system", "content": overall_prompt}]
                )
                
                # Update the overall description
                description = overall_response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error getting overall sentiment: {str(e)}")
            # Keep the last description if error occurs
    
    # Determine the priority
    priority = "facial_expression" if is_fer_valid else "sentiment"
    if not is_fer_valid and not has_sentiment_data:
        priority = "none"
    
    return {
        "priority": priority,
        "facial_expression": facial_expression if isinstance(facial_expression, str) else "unknown",
        "sentiment_analysis": detailed_results,
        "accumulative_sentiment": description,
        "impression_score": impression_score
    }
