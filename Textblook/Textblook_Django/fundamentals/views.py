from django.views.generic import CreateView, UpdateView, TemplateView
from dal import autocomplete
from django.urls import reverse_lazy
#Forms
from .forms import ProfileCreateForm, test_signup
from book.forms import TextbookAutoCompleteForm
#Mixins
from django.contrib.auth.mixins import LoginRequiredMixin
#Imported Models
from fundamentals.models import Profile
from sellingPost.models import SellingPost
from django.contrib.auth.models import User
from book.models import Textbook
#Views
class ProfileCreateView(LoginRequiredMixin, CreateView):
    form_class = ProfileCreateForm
    template_name = 'fundamentals/profileForm.html'
    success_url = '/'

    def form_valid(self, form):
        instance = form.save(commit=False)
        instance.user = self.request.user
        return super(ProfileCreateView, self).form_valid(form)

    def get_context_data(self, *args, **kwargs):
        context = super(ProfileCreateView, self).get_context_data(*args, **kwargs)
        return context

    def get_queryset(self):
        return Profile.objects.filter(user=self.request.user)

class ProfileUpdateView(LoginRequiredMixin, UpdateView):
    form_class = ProfileCreateForm
    template_name = 'fundamentals/profileFormUpdate.html'
    success_url = '/'

    def get_context_data(self, *args, **kwargs):
        user_profile = Profile.objects.get(user=self.request.user)
        user_posts = user_profile.sellingpost_set.all
        context = super(ProfileUpdateView, self).get_context_data(*args, **kwargs)
        context['users_posts'] = user_posts
        return context

    def get_queryset(self):
        return Profile.objects.filter(user=self.request.user)

class BookTitleAutocomplete(TemplateView):
    model = Textbook
    form_class = TextbookAutoCompleteForm
    template_name = 'fundamentals/auto.html'
    success_url = reverse_lazy('title-autocomplete')

    def get_context_data(self, *args, **kwargs):
        context = super(BookTitleAutocomplete, self).get_context_data(*args, **kwargs)
        context['form'] = TextbookAutoCompleteForm()
        print(context)
        return context
