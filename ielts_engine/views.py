from django.shortcuts import render, redirect, get_object_or_404
from .models import Essay, Evaluation, Student
from .forms import EssaySubmissionForm, StudentForm
from django.db.models import Count, Q, Max, Subquery, OuterRef
import io
import json
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
            essay.student = form.cleaned_data['student']
            
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
                from .analytics.master_predict import predict_ensemble
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
                    feedback_comments=results['llm_scores']['feedback_summary'],
                    shap_data=results.get('shap_data', [])
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
        'shap_data_json': json.dumps(ai_eval.shap_data) if ai_eval and ai_eval.shap_data else "[]",
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
        def parse_score(val):
            return float(val) if val and str(val).strip() else None

        evaluation.task_achievement = parse_score(request.POST.get('task_achievement'))
        evaluation.coherence_cohesion = parse_score(request.POST.get('coherence_cohesion'))
        evaluation.lexical_resource = parse_score(request.POST.get('lexical_resource'))
        evaluation.grammar_accuracy = parse_score(request.POST.get('grammar_accuracy'))
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
        
        if eval and eval.overall_band is not None:
            progress_data.append({
                'date': essay.submission_date.strftime('%Y-%m-%d'),
                'score': float(eval.overall_band),
                'ta_score': float(eval.task_achievement) if eval.task_achievement else None,
                'cc_score': float(eval.coherence_cohesion) if eval.coherence_cohesion else None,
                'lr_score': float(eval.lexical_resource) if eval.lexical_resource else None,
                'gra_score': float(eval.grammar_accuracy) if eval.grammar_accuracy else None,
                'task': essay.task_type,
                'essay_id': essay.id
            })
            
    context = {
        'student': student,
        'progress_data': progress_data,
        'progress_data_json': json.dumps(progress_data),
    }
    return render(request, 'ielts_engine/student_progress.html', context)

# ============================================================
# Student CRUD Views
# ============================================================

def student_list(request):
    search_query = request.GET.get('q', '').strip()
    
    students = Student.objects.all().order_by('-enrollment_date')
    
    if search_query:
        students = students.filter(
            Q(name__icontains=search_query) |
            Q(student_id__icontains=search_query) |
            Q(email__icontains=search_query)
        )
    
    # Annotate each student with essay count and latest overall band
    students = students.annotate(
        essay_count=Count('essays')
    )
    
    # Build student data with latest score
    student_data = []
    for student in students:
        latest_eval = Evaluation.objects.filter(
            essay__student=student
        ).order_by('-created_at').first()
        
        latest_score = float(latest_eval.overall_band) if latest_eval and latest_eval.overall_band else None
        
        student_data.append({
            'student': student,
            'essay_count': student.essay_count,
            'latest_score': latest_score,
        })
    
    context = {
        'student_data': student_data,
        'search_query': search_query,
    }
    return render(request, 'ielts_engine/student_list.html', context)

def student_add(request):
    if request.method == 'POST':
        form = StudentForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('student_list')
    else:
        form = StudentForm()
    
    return render(request, 'ielts_engine/student_form.html', {
        'form': form,
        'title': 'Add New Student',
        'submit_label': 'Create Student',
    })

def student_edit(request, student_id):
    student = get_object_or_404(Student, id=student_id)
    
    if request.method == 'POST':
        form = StudentForm(request.POST, instance=student)
        if form.is_valid():
            form.save()
            return redirect('student_list')
    else:
        form = StudentForm(instance=student)
    
    return render(request, 'ielts_engine/student_form.html', {
        'form': form,
        'student': student,
        'title': f'Edit Student — {student.student_id}',
        'submit_label': 'Save Changes',
    })

def student_delete(request, student_id):
    student = get_object_or_404(Student, id=student_id)
    if request.method == 'POST':
        student.delete()
        return redirect('student_list')
    # If GET, redirect back to list (shouldn't happen via UI)
    return redirect('student_list')
