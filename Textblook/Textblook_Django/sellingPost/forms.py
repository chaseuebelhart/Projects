from sellingPost.models import SellingPost
from django import forms
from django.forms import TextInput, Textarea, Select

class SellingPostCreateForm(forms.ModelForm):
    class Meta:
        model = SellingPost
        fields = [
            'sellingPrice',
            'description',
            'condition',
        ]
        widgets = {
            'sellingPrice': TextInput(attrs={'class':'form-control', 'placeholder':'100.00'}),
            'description': Textarea(attrs={'class':'form-control', 'placeholder':'Enter your description...'}),
            'condition': Select(attrs={'class':'form-control'}),
        }
        error_messages = {
        'sellingPrice': {'invalid': "Please enter a number in this range: 0.01 to 9999.99", 'max_digits':"Please enter a number in this range: 0.01 to 9999.99"},
        }
