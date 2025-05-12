# utils/bag_of_words.py

import re
class BagOfWordsFilter:
    def __init__(self, bag_of_words):
        """
        Initialize the BagOfWordsFilter with a bag of words.
        Args:
            bag_of_words (list): A list of words or phrases to check against.
        """
        self.bag_of_words = set(word.lower() for word in bag_of_words)  # Normalize to lowercase

    def check_sentence(self, sentence):
        """
        Check if each word from the bag of words is in the input sentence.
        Args:
            sentence (str): Input sentence to check.
        Returns:
            dict: A dictionary with each word as a key and its presence (True/False) as the value.
        """
        # Normalize the sentence
        normalized_sentence = sentence.lower()

        # Check each word in the bag of words
        return {word: bool(re.search(r'\b' + re.escape(word) + r'\b', normalized_sentence)) for word in self.bag_of_words}