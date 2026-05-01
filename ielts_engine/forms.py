from django import forms
from .models import Essay, Student

class StudentForm(forms.ModelForm):
    class Meta:
        model = Student
        fields = ['name', 'email', 'target_band_score', 'notes']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Full Name'}),
            'email': forms.EmailInput(attrs={'class': 'form-control', 'placeholder': 'email@example.com'}),
            'target_band_score': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.5', 'min': '0', 'max': '9'}),
            'notes': forms.Textarea(attrs={'class': 'form-control', 'rows': 3, 'placeholder': 'Optional notes about this student...'}),
        }

class EssaySubmissionForm(forms.ModelForm):
    student = forms.ModelChoiceField(
        queryset=Student.objects.all().order_by('name'),
        label="Student",
        widget=forms.Select(attrs={'class': 'form-select'}),
        empty_label="-- Select a Student --"
    )
    uploaded_file = forms.FileField(required=False, widget=forms.FileInput(attrs={'class': 'form-control', 'accept': '.pdf,.doc,.docx'}))
    
    class Meta:
        model = Essay
        fields = ['task_type', 'prompt_text', 'extracted_text']
        widgets = {
            'task_type': forms.Select(attrs={'class': 'form-select'}),
            'prompt_text': forms.Textarea(attrs={'class': 'form-control', 'rows': 3, 'placeholder': 'Enter the question prompt here...'}),
            'extracted_text': forms.Textarea(attrs={'class': 'form-control', 'rows': 15, 'placeholder': 'Write your essay here (or upload a file above)...'}),
        }
        labels = {
            'extracted_text': 'Essay Content'
        }
        
    def __init__(self, *args, **kwargs):
        super(EssaySubmissionForm, self).__init__(*args, **kwargs)
        self.fields['extracted_text'].required = False
        # Refresh queryset on each form instantiation
        self.fields['student'].queryset = Student.objects.all().order_by('name')
