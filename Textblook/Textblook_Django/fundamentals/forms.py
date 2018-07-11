from fundamentals.models import Profile
from django import forms
from django.forms import TextInput

class ProfileCreateForm(forms.ModelForm):
    class Meta:
        model = Profile
        fields = [
            'facebook_username',
            'university_email',
        ]
        widgets = {
            'facebook_username': TextInput(attrs={'class':'form-control', 'placeholder':'Facebook Username'}),
            'university_email': TextInput(attrs={'class':'form-control', 'placeholder':'University Email'}),
        }
        error_messages = {
        }

    def clean_university_email(self):
        email = self.cleaned_data['university_email']
        domain = email.split("@")[1]
        if domain != 'umn.edu':
            raise forms.ValidationError("You must enter your umn.edu email")
        return email

class test_signup(forms.Form):
    facebook_username = forms.CharField(widget=forms.TextInput(attrs={'class':'form-control', 'placeholder':'Facebook Username'}))

    def signup(self, request, user):
        email = self.cleaned_data['email']
        user.email = email
        facebook_username = self.cleaned_data['facebook_username']
        university_email = self.cleaned_data['email']
        Profile.objects.create(user = user, facebook_username = facebook_username, university_email = university_email)
        user.save()
