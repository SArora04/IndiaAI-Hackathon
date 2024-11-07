from django.shortcuts import render

# Create your views here.
# complaint/views.py

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import Complaint
from .nlp_model import category_classification,sub_category_classification

@csrf_exempt  # For simplicity, you can add CSRF protection if you're using forms
def classify_complaint_view(request):
    if request.method == 'POST':
        complaint_text = request.POST.get('complaint_text')
        print("Complain : ", complaint_text)

        if complaint_text:
            # Classify the complaint using the NLP model
            prediction_modelResult = category_classification(complaint_text)
            category_modelResult = sub_category_classification(complaint_text,prediction_modelResult)
            # Save the complaint to the database
            #complaint = Complaint.objects.create(complaint_text=complaint_text, category=category)

            # Return the result as JSON
            return JsonResponse({'status': 'success', 'category': category_modelResult, 'sub-category': 'NA'})

        return JsonResponse({'status': 'error', 'message': 'No complaint text provided.'})

    return render(request, 'form.html')
