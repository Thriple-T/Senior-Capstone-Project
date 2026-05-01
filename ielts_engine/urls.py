from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('submit/', views.submit_essay, name='submit_essay'),
    path('evaluation/<int:essay_id>/', views.evaluation_view, name='evaluation_view'),
    path('override/<int:essay_id>/', views.submit_override, name='submit_override'),
    path('student/progress/<int:student_id>/', views.student_progress, name='student_progress'),
    
    # Student CRUD
    path('students/', views.student_list, name='student_list'),
    path('students/add/', views.student_add, name='student_add'),
    path('students/<int:student_id>/edit/', views.student_edit, name='student_edit'),
    path('students/<int:student_id>/delete/', views.student_delete, name='student_delete'),
]
