from django.views.generic import CreateView, UpdateView, DeleteView
from django.core.urlresolvers import reverse_lazy
from django.http import HttpResponseForbidden
#Forms
from sellingPost.forms import SellingPostCreateForm
#Mixins
from django.contrib.auth.mixins import LoginRequiredMixin
#Models
from sellingPost.models import SellingPost
from django.contrib.auth.models import User

class SellingPostCreateView(LoginRequiredMixin, CreateView):
    form_class = SellingPostCreateForm
    template_name = 'sellingPost/sellForm.html'
    success_url = '/'

    def form_valid(self, form):
        instance = form.save(commit=False)
        instance.user = self.request.user
        return super(SellingPostCreateView, self).form_valid(form)

    def get_context_data(self, *args, **kwargs):
        context = super(SellingPostCreateView, self).get_context_data(*args, **kwargs)
        return context

class SellingPostUpdateView(LoginRequiredMixin, UpdateView):
    form_class = SellingPostCreateForm
    model = SellingPost
    template_name = 'sellingPost/sellForm.html'
    success_url = '/'

    def get_context_data(self, *args, **kwargs):
        context = super(SellingPostUpdateView, self).get_context_data(*args, **kwargs)
        return context


class SellingPostDeleteView(LoginRequiredMixin, DeleteView):
    model = SellingPost

    def get_success_url(self):
        return reverse_lazy('book:book-listing', kwargs={'slug': self.object.textbook.slug})

    def post(self, request, *args, **kwargs):
        if not request.user.is_authenticated:
            return HttpResponseForbidden()
        self.object = self.get_object()
        if self.object.profile.user == request.user:
            return self.delete(request, *args, **kwargs)
        else:
            return HttpResponseForbidden()
