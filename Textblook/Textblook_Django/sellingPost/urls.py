from django.conf.urls import url, include
from . import views
from sellingPost.views import SellingPostCreateView, SellingPostUpdateView

urlpatterns = [
    url(r'^textbook/create/$', SellingPostCreateView.as_view()),
    ]
