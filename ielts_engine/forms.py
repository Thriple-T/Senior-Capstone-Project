from django import forms
from .models import Essay, Student

class EssaySubmissionForm(forms.ModelForm):
    student_identifier = forms.CharField(
        label="Student Name or ID",
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Enter Student Name or ID...'})
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
