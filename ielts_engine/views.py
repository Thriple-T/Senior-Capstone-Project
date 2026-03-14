from django.shortcuts import render, redirect, get_object_or_404
from .models import Essay, Evaluation, Student
from .forms import EssaySubmissionForm
from .analytics.master_predict import predict_ensemble
import io
import pypdf
from docx import Document

def dashboard(request):
    # Fetch recent evaluations (AI) for the dashboard
    evaluations = Evaluation.objects.filter(evaluator_type='AI').select_related('essay__student').order_by('-created_at')[:10]
    students = Student.objects.all()
    
    total_score = 0
    scored_count = 0
    support_needs = set()
    
    for ev in evaluations:
        if ev.overall_band:
            total_score += float(ev.overall_band)
            scored_count += 1
            if float(ev.overall_band) < 6.0:
                support_needs.add(ev.essay.student)
                
    class_mastery = round(total_score / scored_count, 1) if scored_count > 0 else 0

    context = {
        'evaluations': evaluations,
        'students': students,
        'class_mastery': class_mastery,
        'support_needs': list(support_needs),
    }
    return render(request, 'ielts_engine/dashboard.html', context)

def submit_essay(request):
    if request.method == 'POST':
        form = EssaySubmissionForm(request.POST, request.FILES)
        
        if form.is_valid():
            essay = form.save(commit=False)
            
            # Handle Student Lookup or Creation by identifier (Name or ID)
            identifier = form.cleaned_data['student_identifier']
            student = None
            
            # Try lookup by ID first if numeric
            if identifier.isdigit():
                student = Student.objects.filter(id=int(identifier)).first()
            
            # Fallback to lookup by Name
            if not student:
                student = Student.objects.filter(name__iexact=identifier).first()
            
            # Create if not found (Simple implementation)
            if not student:
                student = Student.objects.create(name=identifier)
            
            essay.student = student
            
            # Handle uploaded file extraction
            uploaded_file = request.FILES.get('uploaded_file')
            if uploaded_file:
                extracted_text = ""
                file_name = uploaded_file.name.lower()
                
                try:
                    if file_name.endswith('.pdf'):
                        pdf_reader = pypdf.PdfReader(io.BytesIO(uploaded_file.read()))
                        for page in pdf_reader.pages:
                            text = page.extract_text()
                            if text:
                                extracted_text += text + "\n"
                    elif file_name.endswith('.docx') or file_name.endswith('.doc'):
                        doc = Document(io.BytesIO(uploaded_file.read()))
                        for para in doc.paragraphs:
                            extracted_text += para.text + "\n"
                except Exception:
                    pass
                
                if extracted_text.strip():
                    essay.extracted_text = extracted_text.strip()
            
            essay.save()
            
            # Create AI Evaluation using the REAL AI Ensemble Platform
            try:
                # The user form doesn't seem to have a prompt input yet, default to a Task 2 generic string 
                # if prompt_text is empty, otherwise use the saved prompt_text 
                engine_prompt = essay.prompt_text if essay.prompt_text else "Write an essay discussing your opinion on this topic."
                
                results = predict_ensemble(essay.extracted_text, engine_prompt)
                
                Evaluation.objects.create(
                    essay=essay,
                    evaluator_type='AI',
                    task_achievement=results['ensemble_scores']['Task_Achievement'],
                    coherence_cohesion=results['ensemble_scores']['Coherence_Cohesion'],
                    lexical_resource=results['ensemble_scores']['Lexical_Resource'],
                    grammar_accuracy=results['ensemble_scores']['Grammar_Range'],
                    overall_band=results['ensemble_scores']['Overall_Band'],
                    feedback_comments=results['llm_scores']['feedback_summary']
                )
            except Exception as e:
                # Fallback safeguard in case AI pipeline crashes
                Evaluation.objects.create(
                    essay=essay,
                    evaluator_type='AI',
                    overall_band=0.0,
                    feedback_comments=f"AI Engine failed to process this essay. Error: {str(e)}"
                )
            
            return redirect('evaluation_view', essay_id=essay.id)
    else:
        form = EssaySubmissionForm()
    
    return render(request, 'ielts_engine/submission_form.html', {'form': form})

def evaluation_view(request, essay_id):
    essay = get_object_or_404(Essay, id=essay_id)
    
    # Get evaluations
    ai_eval = essay.evaluations.filter(evaluator_type='AI').first()
    teacher_eval = essay.evaluations.filter(evaluator_type='TEACHER').first()
    
    # Active evaluation for UI defaults
    evaluation = teacher_eval if teacher_eval else ai_eval
    
    context = {
        'essay': essay,
        'ai_eval': ai_eval,
        'teacher_eval': teacher_eval,
        'evaluation': evaluation,  # Current active one
        'errors': [], # The real evaluator currently does not return specific char-level error positions.
    }
    return render(request, 'ielts_engine/evaluation.html', context)

def submit_override(request, essay_id):
    if request.method == 'POST':
        essay = get_object_or_404(Essay, id=essay_id)
        
        # Get or create Teacher evaluation
        evaluation, created = Evaluation.objects.get_or_create(
            essay=essay,
            evaluator_type='TEACHER'
        )
        
        # Update scores from post data
        evaluation.task_achievement = request.POST.get('task_achievement')
        evaluation.coherence_cohesion = request.POST.get('coherence_cohesion')
        evaluation.lexical_resource = request.POST.get('lexical_resource')
        evaluation.grammar_accuracy = request.POST.get('grammar_accuracy')
        evaluation.feedback_comments = request.POST.get('feedback_comments', '')
        
        # Re-calculate overall band (handled in model save)
        evaluation.overall_band = None 
        evaluation.save()
        
        return redirect('evaluation_view', essay_id=essay.id)
    
def student_progress(request, student_id):
    student = get_object_or_404(Student, id=student_id)
    essays = student.essays.all().order_by('submission_date')
    
    progress_data = []
    for essay in essays:
        # Get the 'best' evaluation for this essay (Teacher override if exists, else AI)
        eval = essay.evaluations.filter(evaluator_type='TEACHER').first() or \
               essay.evaluations.filter(evaluator_type='AI').first()
        
        if eval:
            progress_data.append({
                'date': essay.submission_date.strftime('%Y-%m-%d'),
                'score': float(eval.overall_band),
                'task': essay.task_type,
                'essay_id': essay.id
            })
            
    context = {
        'student': student,
        'progress_data': progress_data,
    }
    return render(request, 'ielts_engine/student_progress.html', context)
