from django.conf.urls import url, include
from . import views
from django.views.generic import TemplateView
from .views import TextbookListView, TextbookDetailView
from sellingPost.views import SellingPostUpdateView, SellingPostDeleteView

urlpatterns = [
    url(r'^(?P<slug>[\w-]+)$', TextbookDetailView.as_view(), name='book-listing'),
    url(r'^(?P<slug>[\w-]+)/(?P<pk>\d+)/edit/$', SellingPostUpdateView.as_view(), name ='update-post'),
    url(r'^$', TextbookListView.as_view(), name='books-list'),
    url(r'^(?P<slug>[\w-]+)/(?P<pk>\d+)/delete$', SellingPostDeleteView.as_view(), name='delete-post'),
    ]
