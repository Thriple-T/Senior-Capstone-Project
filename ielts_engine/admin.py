from django.contrib import admin
from .models import TeacherProfile, Student, Essay, Evaluation

@admin.register(TeacherProfile)
class TeacherProfileAdmin(admin.ModelAdmin):
    list_display = ('user',)

@admin.register(Student)
class StudentAdmin(admin.ModelAdmin):
    list_display = ('name', 'target_band_score', 'enrollment_date')
    search_fields = ('name',)

@admin.register(Essay)
class EssayAdmin(admin.ModelAdmin):
    list_display = ('student', 'task_type', 'submission_date')
    list_filter = ('task_type',)

@admin.register(Evaluation)
class EvaluationAdmin(admin.ModelAdmin):
    list_display = ('essay', 'evaluator_type', 'overall_band', 'created_at')
    list_filter = ('evaluator_type',)
