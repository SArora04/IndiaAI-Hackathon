# complaint/models.py

from django.db import models

class Complaint(models.Model):
    complaint_text = models.TextField()  # The text of the complaint
    category = models.CharField(max_length=50, null=True, blank=True)  # The predicted category (optional)
    # include sub-category

    def __str__(self):
        return f"Complaint ID: {self.id}, Category: {self.category}"
