from django.shortcuts import render

# Create your views here.
# complaint/views.py

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import Complaint
from .nlp_model import category_classification,subcategory_classification,clean_text

@csrf_exempt  # For simplicity, you can add CSRF protection if you're using forms
def classify_complaint_view(request):
    if request.method == 'POST':
        complaint_text = request.POST.get('complaint_text')
        # print("Complain : ", complaint_text)

        if complaint_text:
            # Classify the complaint using the NLP model
            cleaned_complain_text = clean_text(complaint_text)
            prediction_modelResult = category_classification(cleaned_complain_text)
            category_modelResult, subcategory_modelResult = subcategory_classification(cleaned_complain_text,prediction_modelResult)

            # Return the result as JSON
            return JsonResponse({'status': 'success', 'category': category_modelResult, 'sub-category': subcategory_modelResult})

        return JsonResponse({'status': 'error', 'message': 'No complaint text provided.'})

    return render(request, 'form.html')
