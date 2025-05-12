
# Callpilot-AI

This project provides a robust solution for transcribing audio, analyzing sentiments, and handling multimedia files. It integrates OpenAI's Whisper API and a sentiment analysis model to deliver seamless functionality for processing video files, converting them to audio, generating transcriptions, and extracting emotions from text.

---

## Features

1. **Video to Audio Conversion**: Converts uploaded video files into audio for further processing.
2. **Whisper API Integration**: Uses OpenAI's Whisper API for transcription of audio files.
3. **Q&A Generation**: Converts transcription into a structured Q&A format.
4. **Sentiment Analysis**: Analyzes human responses in transcription for emotional insights using a pre-trained model.
5. **Keyword Detection in Human Responses**: Identifies specific user-defined keywords in human responses from the transcription.

---

## Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/Cplus-Soft-Limited/callpilot-ai.git
cd callpilot-ai
```

### Step 2: Create a Virtual Environment
```bash
python -m venv .venv
```
Activate the virtual environment:
- On **Windows**:
  ```bash
  .venv\Scripts\activate
  ```
- On **Linux/Mac**:
  ```bash
  source .venv/bin/activate
  ```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Set Up Environment Variables
Create a `.env` file in the root directory and add your OpenAI API key:
```
OPENAI_API_KEY=your_openai_api_key
```

---


## File Structure

- **`app.py`**: FastAPI application for handling video uploads, transcriptions, and sentiment analysis.
- **`wisperapi.py`**: Handles transcription using OpenAI's Whisper API.
- **`emotion_analyzer.py`**: Performs sentiment analysis using a pre-trained emotion model.
- **`requirements.txt`**: Lists all dependencies.

---

## API Endpoints

### 1. `/transcribe` (POST)
Converts a video file to audio, transcribes it, and returns the transcription in a Q&A format.

#### Request
**Input**: Video file

**Example `curl` Command:**
```bash
curl -X POST "http://localhost:8000/transcribe" -H "Content-Type: multipart/form-data" -F "file=@D:/cplus bot/data_video/909.mp4"
```

#### Response
**Output**: JSON object containing the transcription in Q&A format.

**Example Response:**
```json
{
  "id": "example_video",
  "transcription": [
    {"bot": "What is your name?", "human": "John Doe."},
    {"bot": "How are you feeling today?", "human": "I'm feeling good."}
  ]
}
```

### 2. `/update-sentiment` (POST)
Set the custom sentiments.

#### Request
**Input**: JSON

**Example `curl` Command:**
```bash
curl --location 'http://localhost:8000/update-sentiment' --header 'Content-Type: application/json' --data '{
  "happy": "Feeling joy or pleasure.",
  "sad": "Feeling sorrow or unhappiness.",
  "fear": "Feeling feared for upcoming event or action"
}'
```

#### Response
**Output**: JSON object containing the message of setting sentiments.

**Example Response:**
```json
{
    "message": "Sentiment definitions updated successfully."
}
```

### 3. `/analyze-sentiment-with-custom` (POST)
Analyzes the sentiment of human responses in the transcription.

#### Request
**Input**: JSON object containing an `id` and a list of Q&A pairs.

**Example `curl` Command:**
```bash
curl --location 'http://localhost:8000/analyze-sentiment-with-custom' --header 'Content-Type: application/json' --data '{
    "id": "WELCOME_ENGLISH-16563-176037",
    "transcription": [
        {
            "bot": "Hey, there, welcome to the glide solar family and congratulations on your decision to go solar.",
            "human": ""
        },
        {
            "bot": "As I walk you through this process, we will be recording all along.",
            "human": ""
        }
    ]
}'
```

#### Response
**Output**: JSON object containing sentiment scores.

**Example Response:**
```json
{
    "id": "WELCOME_ENGLISH-16563-176037",
    "analysis": [
        {
            "question": "Hey, there, welcome to the glide solar family and congratulations on your decision to go solar.",
            "answer": "none",
            "sentiment": "none"
        },
        {
            "question": "As I walk you through this process, we will be recording all along.",
            "answer": "none",
            "sentiment": "none"
        }
    ]
}
```

### 4. `/check-human-words` (POST)
Identifies specific user-defined keywords in human responses from the transcription.

#### Request
**Input**: JSON object containing an `id`, transcription data, and a list of words to check.

**Example Input**:
```json
{
    "id": "923",
    "transcription": [
        {
            "bot": "The following are a series of questions to ensure that this transaction proceeds in accordance with your expectations and understanding.",
            "human": ""
        },
        {
            "bot": "Do you own your home?",
            "human": "Yes."
        }
    ],
    "words_to_check": ["yes", "no", "home", "agreement", "expectations"]
}
```

**Example `curl` Command:**
```bash
curl -X POST "http://localhost:8000/check-human-words" -H "Content-Type: application/json" -d '{
  "data": {
    "id": "example_id",
    "transcription": [
      {"bot": "How are you?", "human": "I am fine."},
      {"bot": "What is your name?", "human": "John."}
    ],
    "words_to_check": ["fine", "name"]
  },
  "scenarios": ["detect positive tone", "find keywords"]
}'
```

#### Response
**Output**: JSON object containing the presence of each word in the human responses.

**Example Output**:
```json
{
    "id": "923",
    "words_found": {
        "yes": true,
        "no": true,
        "home": false,
        "agreement": false,
        "expectations": false
    }
}
```
### 5. `/transcribe-media` (POST)
Converts a video file to audio, transcribes it, and returns the transcription in a Q&A format.
using Assembly AI
#### Request
**Input**: Video file

**Example `curl` Command:**
```bash
curl -X POST "http://localhost:8000/transcribe-media" -H "Content-Type: multipart/form-data" -F "file=@D:/cplus bot/data_video/909.mp4"
```

#### Response
**Output**: JSON object containing the transcription in Q&A format.

**Example Response:**
```json
{
  "id": "example_video",
  "transcription": [
    {
            "timestamp_start": "0.16 s",
            "timestamp_end": "3.38 s",
            "speaker": "bot",
            "text": "Appears on the screen below your correct billing address.",
            "bot": "Appears on the screen below your correct billing address.",
            "human": "121. RPO box 121."
        },
  ]
}
```
### 6. /process_video` (POST)
do extract human frames from video of perticular time frame do facial expression  and return output 
#### Request
**Input**: Video file

**Example `curl` Command:**
```bash
curl -X POST "http://127.0.0.1:8000/process_video/" \
  -H "Content-Type: multipart/form-data" \
  -F "video=@D:\data_video\907.mp4" \
  -F "start_time=0.16" \
  -F "end_time=3.38"

```

#### Response
**Output**: JSON object containing the transcription in Q&A format.

**Example Response:**
```json
{
    "expressions": [
        "neutral"
    ]
}
```
### 7. /process-data/` (POST)
Processes both sentiment analysis and facial expression data and returns a result with priority, facial expression, sentiment analysis match type, and a final score.
#### Request
**Input**:  JSON object containing sentiment data and facial data.

**Example `curl` Command:**
```bash
curl -X POST "http://127.0.0.1:8000/process-data/" \
  -H "Content-Type: application/json" \
  -d '{
        "sentiment_data": {
          "id": "907",
          "responses_with_sentiment": [
            {
              "timestamp_start": "",
              "timestamp_end": "",
              "question": "Is the billing address that appears on the screen below your correct billing address?",
              "answer": "Yes.",
              "sentiment": "Neutral"
            }
          ]
        },
        "facial_data": {
          "expressions": "happy"
        }
      }'


```

#### Response
**Output**: JSON object containing the result with priority, facial expression, sentiment analysis with match type, and final score.

**Example Response:**
```json
{
  "result": {
    "priority": "facial_expression",
    "facial_expression": "happy",
    "sentiment_analysis": [
      {
        "question": "Is the billing address that appears on the screen below your correct billing address?",
        "answer": "Yes.",
        "sentiment": "Neutral",
        "match_type": "Opposite meaning"
      }
    ],
    "final_score": 30
  }
}
```

### 8. /process_video_1` (POST)
do extract human frames from video of perticular time frames of all human responses in video do facial expression  and return outputs
#### Request
**Input**: Video file

**Example `curl` Command:**
```bash
curl -X POST "http://127.0.0.1:8000/process_video_1/" \
-H "Content-Type: multipart/form-data" \
-F "video=@path_to_your_video/video.mp4" \
-F "transcription={
  \"transcription\": [
    {
      \"timestamp_start\": \"7.60 s\",
      \"timestamp_end\": \"9.67 s\",
      \"speaker\": \"bot\",
      \"text\": \"Appears on the screen below your correct billing address.\",
      \"bot\": \"Appears on the screen below your correct billing address.\",
      \"human\": \"121. RPO box 121.\"
    },
    {
      \"timestamp_start\": \"35.99 s\",
      \"timestamp_end\": \"36.79 s\",
      \"speaker\": \"bot\",
      \"text\": \"Is the billing address that appears on the screen below your correct billing address?\",
      \"bot\": \"Is the billing address that appears on the screen below your correct billing address?\",
      \"human\": \"Yes.\"
    }
  ]
}"


```

### 9./process-data_final/` (POST)
do take sentiment and FER input of whole video do parsing have give output of whole video
#### Request
**Input**: json

**Example `curl` Command:**
```bash
curl -X POST "http://127.0.0.1:8000/process-data_final/" \
-H "Content-Type: application/json" \
-d '{
  "sentiment_data": {
    "id": "907",
    "responses_with_sentiment": [
      {
        "timestamp_start": "",
        "timestamp_end": "",
        "question": "Appears on the screen below your correct billing address.",
        "answer": "121. RPO box 121.",
        "sentiment": "happy"
      },
      {
        "timestamp_start": "",
        "timestamp_end": "",
        "question": "Is the billing address that appears on the screen below your correct billing address?",
        "answer": "Yes.",
        "sentiment": "happy"
      }
    ]
  },
  "facial_data": {
    "expressions": [
      {
        "start_time": "7.60 s",
        "end_time": "9.67 s",
        "error": "No faces detected in the top-left corner of the video."
      },
      {
        "start_time": 35.99,
        "end_time": 36.79,
        "expressions": "happy"
      }
    ]
  }
}'
```
### 10. `/process-video-data` (POST)
Converts a video file to audio, transcribes it, do sentiment anlysis , do FER and at the end do commulative sentiment and return sentiment analysis , FER ,and commulative sentiment
#### Request
**Input**: Video file

**Example `curl` Command:**
```bash
curl -X POST "http://localhost:8000/process-video-data" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@C:\path\to\your\video.mp4"
```

#### Response
**Output**: JSON object containing sentiment anlysis , do FER and at the end do commulative sentiment .

---

## Troubleshooting

1. **Whisper API Errors**:
   - Ensure your OpenAI API key is valid and correctly set in the `.env` file.

2. **Video to Audio Conversion Issues**:
   - Install FFmpeg if audio extraction fails.
     ```bash
     sudo apt-get install ffmpeg
     ```

3. **Sentiment Model Errors**:
   - Ensure the required Hugging Face model is downloaded and accessible.
     ```bash
     pip install transformers
     ```

4. **Keyword Detection Errors**:
   - Ensure the transcription contains valid `human` responses for accurate keyword matching.

---
## Docker Instructions

### Step 1: Build Docker Image
Ensure you are in the root directory of the project where the `Dockerfile` is located. Run the following command to build the Docker image:
```bash
docker build -t callpilot-ai .
```

### Step 2: Run Docker Container
Run the image you just built:
```bash
docker run -p 8000:8000 callpilot-ai
```

This command maps the container's port `8000` to your local machine's port `8000`. You can access the application at `http://localhost:8000`.

### Step 3: Verify Docker Container
You can check the running containers with:
```bash
docker ps
```

To interactively enter the container:
```bash
docker exec -it <container_id> bash
```

Replace `<container_id>` with the actual ID of the running container.

---


## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
