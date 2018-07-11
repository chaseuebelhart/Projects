from django.conf.urls import url, include
from . import views
from django.views.generic import TemplateView
from fundamentals.views import ProfileCreateView, ProfileUpdateView, BookTitleAutocomplete
from allauth.account.views import LogoutView

urlpatterns = [
    url(r'^$', TemplateView.as_view(template_name = 'fundamentals/index.html')),
    url(r'^register/$', ProfileCreateView.as_view()),
    url(r'^editProfile/(?P<slug>[\-\w]+)$', ProfileUpdateView.as_view()),
    url(r'^logout/$', LogoutView.as_view()),
    url(r'^fb$', TemplateView.as_view(template_name = 'fundamentals/profile.html')),
    url(r'^textbooks-search$', TemplateView.as_view(template_name = 'book/textbook_search.html')),
    url(r'^title-autocomplete/$',BookTitleAutocomplete.as_view(), name='title-autocomplete',
    ),
]
