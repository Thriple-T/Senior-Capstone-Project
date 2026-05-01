from django.db import models
from django.contrib.auth.models import User
from django.core.validators import MinValueValidator, MaxValueValidator

class TeacherProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    bio = models.TextField(blank=True, null=True)

    def __str__(self):
        return self.user.username

class Student(models.Model):
    student_id = models.CharField(max_length=20, unique=True, editable=False, blank=True)
    name = models.CharField(max_length=200)
    email = models.EmailField(blank=True, null=True)
    target_band_score = models.DecimalField(max_digits=3, decimal_places=1, default=6.5)
    notes = models.TextField(blank=True, null=True)
    enrollment_date = models.DateTimeField(auto_now_add=True)
    teacher = models.ForeignKey(TeacherProfile, on_delete=models.CASCADE, related_name="students", null=True, blank=True)

    def save(self, *args, **kwargs):
        if not self.student_id:
            last = Student.objects.order_by('-id').first()
            next_num = (last.id + 1) if last else 1
            self.student_id = f"STU-{next_num:04d}"
        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.student_id} - {self.name}"

class Essay(models.Model):
    TASK_CHOICES = [
        (1, 'Task 1 (Academic/General)'),
        (2, 'Task 2 (Essay)'),
    ]

    student = models.ForeignKey(Student, on_delete=models.CASCADE, related_name="essays")
    task_type = models.SmallIntegerField(choices=TASK_CHOICES)
    prompt_text = models.TextField(blank=True, null=True)
    file_path = models.TextField(blank=True, null=True) # or FileField
    extracted_text = models.TextField()
    submission_date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.student.name} - Task {self.task_type} - {self.submission_date.strftime('%Y-%m-%d')}"

class Evaluation(models.Model):
    EVALUATOR_CHOICES = [
        ('AI', 'AI'),
        ('TEACHER', 'Teacher'),
    ]

    essay = models.ForeignKey(Essay, on_delete=models.CASCADE, related_name="evaluations")
    evaluator_type = models.CharField(max_length=10, choices=EVALUATOR_CHOICES)
    
    # Grading Fields (0.0 to 9.0)
    task_achievement = models.DecimalField(max_digits=3, decimal_places=1, null=True, blank=True, validators=[MinValueValidator(0.0), MaxValueValidator(9.0)])
    coherence_cohesion = models.DecimalField(max_digits=3, decimal_places=1, null=True, blank=True, validators=[MinValueValidator(0.0), MaxValueValidator(9.0)])
    lexical_resource = models.DecimalField(max_digits=3, decimal_places=1, null=True, blank=True, validators=[MinValueValidator(0.0), MaxValueValidator(9.0)])
    grammar_accuracy = models.DecimalField(max_digits=3, decimal_places=1, null=True, blank=True, validators=[MinValueValidator(0.0), MaxValueValidator(9.0)])
    overall_band = models.DecimalField(max_digits=3, decimal_places=1, null=True, blank=True)
    
    feedback_comments = models.TextField(blank=True, null=True)
    shap_data = models.JSONField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def calculate_overall(self):
        scores = [self.task_achievement, self.coherence_cohesion, self.lexical_resource, self.grammar_accuracy]
        try:
            valid_scores = [float(s) for s in scores if s is not None and str(s).strip() != '']
            if len(valid_scores) == 4:
                avg = sum(valid_scores) / 4
                # Official IELTS rounding:
                # 6.25 -> 6.5
                # 6.75 -> 7.0
                # 6.125 -> 6.0
                fraction = avg % 1
                if fraction < 0.25:
                    return float(int(avg))
                elif fraction < 0.75:
                    return float(int(avg)) + 0.5
                else:
                    return float(int(avg)) + 1.0
        except (ValueError, TypeError):
            pass
        return None

    def save(self, *args, **kwargs):
        if not self.overall_band:
            self.overall_band = self.calculate_overall()
        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.evaluator_type} - {self.essay}"
