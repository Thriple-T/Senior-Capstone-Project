import re

class IELTSEvaluator:
    def analyze(self, text, task_type='2'):
        """
        Analyzes the essay text and returns basic metrics.
        This is a stub implementation.
        """
        word_count = len(text.split())
        
        # Simple lexical diversity: ratio of unique words to total words
        words = re.findall(r'\b\w+\b', text.lower())
        unique_words = set(words)
        lexical_diversity = len(unique_words) / len(words) if words else 0

        # Simple sentence complexity: heuristic based on sentence length and commas
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        complex_sentences = [s for s in sentences if len(s.split()) > 15 or ',' in s]
        complexity_ratio = len(complex_sentences) / len(sentences) if sentences else 0

        return {
            'word_count': word_count,
            'lexical_diversity': round(lexical_diversity, 2),
            'sentence_complexity_ratio': round(complexity_ratio, 2),
            'is_under_length': self._check_length(word_count, task_type)
        }

    def _check_length(self, count, task_type):
        min_words = 150 if task_type == '1' else 250
        return count < min_words
