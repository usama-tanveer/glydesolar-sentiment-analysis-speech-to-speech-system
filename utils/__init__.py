# transcriber_pkg/__init__.py

from .bag_of_words import BagOfWordsFilter
from .emotion_analyzer import EmotionAnalyzer

__version__ = "1.1.0"
__all__ = [ "BagOfWordsFilter", "EmotionAnalyzer"]
