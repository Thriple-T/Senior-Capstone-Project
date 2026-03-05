class MockEvaluator:
    def __init__(self):
        self.scenarios = {
            'at_risk': {
                'overall_band': 4.5,
                'task_response': 4.5,
                'coherence_cohesion': 4.0,
                'lexical_resource': 4.5,
                'grammatical_range_accuracy': 5.0,
                'errors': [
                    {"type": "vocabulary", "text": "Repetitive use of basic vocabulary ('good', 'bad').", "severity": "high"},
                    {"type": "grammar", "text": "Frequent subject-verb agreement errors across multiple sentences.", "severity": "high"},
                    {"type": "coherence", "text": "Lack of clear paragraphing and transitions.", "severity": "high"},
                    {"type": "task_response", "text": "Under word count (only 120 words).", "severity": "high"}
                ]
            },
            'standard': {
                'overall_band': 6.0,
                'task_response': 6.0,
                'coherence_cohesion': 6.0,
                'lexical_resource': 6.0,
                'grammatical_range_accuracy': 6.0,
                'errors': [
                    {"type": "vocabulary", "text": "Some inappropriate word choices for academic contexts.", "severity": "medium"},
                    {"type": "grammar", "text": "Occasional errors in complex sentence structures.", "severity": "medium"},
                    {"type": "coherence", "text": "Cohesion is somewhat mechanical; overuse of basic linkers.", "severity": "medium"}
                ]
            },
            'high_performing': {
                'overall_band': 8.0,
                'task_response': 8.0,
                'coherence_cohesion': 8.0,
                'lexical_resource': 8.0,
                'grammatical_range_accuracy': 8.0,
                'errors': [
                    {"type": "vocabulary", "text": "Rare minor inaccuracies in collocation.", "severity": "low"},
                    {"type": "grammar", "text": "Occasional, non-systematic slips in grammar.", "severity": "low"}
                ]
            }
        }

    def evaluate(self, text, scenario='standard'):
        """
        Returns mock evaluation results based on the chosen scenario.
        If scenario is not provided, defaults to standard.
        """
        # A simple check for extremely short essays overriding scenario
        word_count = len(text.split())
        if word_count > 0 and word_count < 100 and scenario != 'at_risk':
            scenario = 'at_risk'

        if scenario not in self.scenarios:
            scenario = 'standard'
            
        result = self.scenarios[scenario].copy()
        
        return result
