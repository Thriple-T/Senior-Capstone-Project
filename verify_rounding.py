import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'sayardesk.settings')
django.setup()

from ielts_engine.models import Evaluation

def test_rounding():
    test_cases = [
        (6.0, 6.0, 6.0, 6.0, 6.0),    # Avg 6.0 -> 6.0
        (6.0, 6.5, 6.0, 6.5, 6.5),    # Avg 6.25 -> 6.5
        (6.5, 7.0, 6.5, 7.0, 7.0),    # Avg 6.75 -> 7.0
        (6.0, 6.0, 6.0, 6.5, 6.0),    # Avg 6.125 -> 6.0
        (7.0, 7.0, 7.5, 7.5, 7.5),    # Avg 7.25 -> 7.5
        (8.0, 8.5, 8.5, 8.5, 8.5),    # Avg 8.375 -> 8.5
    ]
    
    print("Testing IELTS Rounding Logic:")
    for ta, cc, lr, gra, expected in test_cases:
        eval = Evaluation(
            task_achievement=ta,
            coherence_cohesion=cc,
            lexical_resource=lr,
            grammar_accuracy=gra
        )
        actual = eval.calculate_overall()
        status = "PASSED" if actual == expected else f"FAILED (Expected {expected}, Got {actual})"
        print(f"Scores: [{ta}, {cc}, {lr}, {gra}] -> Avg: {(ta+cc+lr+gra)/4} -> Result: {actual} | {status}")

if __name__ == "__main__":
    test_rounding()
