from deepgram import DeepgramClient, PrerecordedOptions
import re

# The API key we created in step 3
DEEPGRAM_API_KEY = '611c1a51395ee1c2560ccb66356b5d5b9cb69ac2'

# Replace with your file path
PATH_TO_FILE = '1.mp3'

def remove_punctuation(text):
    # Remove punctuation from the text
    return re.sub(r'[^\w\s]', '', text)

def main():
    deepgram = DeepgramClient(DEEPGRAM_API_KEY)

    with open(PATH_TO_FILE, 'rb') as audio:
        payload = {'buffer': audio}

        options = PrerecordedOptions(
            smart_format=True,
            model="nova-3",
            language="en-US",
            diarize=True  # Enable speaker diarization
        )

        # For larger files, you might need to increase the timeout
        response = deepgram.listen.prerecorded.v('1').transcribe_file(payload, options)

        # Print the full response to check if 'speaker_labels' is present
        print("Full Response:")
        print(response.to_json(indent=4))

        try:
            # Extract the transcription and speaker information
            transcription = response['results']['channels'][0]['alternatives'][0]['transcript']
            word_info_list = response['results']['channels'][0]['alternatives'][0]['words']

            conversation = []
            current_speaker = None
            sentence = []

            for word_info in word_info_list:
                word = word_info['word']
                speaker = word_info['speaker']

                # Remove punctuation from the word
                word = remove_punctuation(word)

                if speaker != current_speaker and sentence:
                    # Append the sentence for the previous speaker
                    conversation.append(f"{'bot' if current_speaker == 1 else 'human'}: {' '.join(sentence)}")
                    sentence = []  # Reset the sentence for the new speaker

                # Add the word to the sentence
                sentence.append(word)
                current_speaker = speaker

            # Append the last sentence
            if sentence:
                conversation.append(f"{'bot' if current_speaker == 1 else 'human'}: {' '.join(sentence)}")

            # Print the conversation
            print("\nConversation:")
            print("\n".join(conversation))
        
        except KeyError as e:
            print(f"Error: Missing key {e}. Response structure might be different.")
            print("Please check the full response above to adjust parsing.")

if __name__ == '__main__':
    main()
