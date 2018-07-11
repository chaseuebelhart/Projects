from book.models import Textbook
from django import forms
from dal import autocomplete

class TextbookAutoCompleteForm(forms.ModelForm):
    class Meta:
        model = Textbook
        fields = [
            'title',
            'author',
        ]
        widgets = {
            'title': autocomplete.ModelSelect2(url='title-autocomplete',
                                               attrs = {
                                                    # Set some placeholder
                                                    'data-placeholder': 'Autocomplete ...',
                                                    # Only trigger autocompletion after 3 characters have been typed
                                                    'data-minimum-input-length': 3,
                                                }
                                                ),
        }