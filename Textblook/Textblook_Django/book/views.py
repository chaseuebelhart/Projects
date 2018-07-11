from django.views.generic import ListView, DetailView
from django.http import HttpResponseForbidden
from django.shortcuts import get_object_or_404, render
from django.core.urlresolvers import reverse_lazy
from django.db.models import Q
#Forms
from book.forms import TextbookAutoCompleteForm
from sellingPost.forms import SellingPostCreateForm
#Mixins
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic.edit import FormMixin
#Imported Models
from book.models import Textbook
from sellingPost.models import SellingPost
from allauth.socialaccount.models import SocialAccount
#Views
class TextbookListView(ListView):
    def get_queryset(self):
        queryset_list = Textbook.objects.none()
        query = self.request.GET.get('q')
        if query:
            queryset_list = Textbook.objects.all()
            queryset_list = queryset_list.filter(
                Q(title__icontains=query)|
                Q(author__icontains=query)|
                Q(isbn__icontains=query)|
                Q(klass__icontains=query)
                )
        return queryset_list

class TextbookDetailView(LoginRequiredMixin, FormMixin, DetailView):
    queryset = Textbook.objects.all()
    model = SellingPost
    form_class = SellingPostCreateForm

    def post(self, request, *args, **kwargs):
        self.object = self.get_object()
        form = self.get_form()
        if form.is_valid():
            sellingPrice = form.cleaned_data['sellingPrice']
            description = form.cleaned_data['description']
            condition = form.cleaned_data['condition']
            textbook_slug = kwargs['slug']
            textbook = Textbook.objects.get(slug = textbook_slug)
            SellingPost.objects.create(profile = self.request.user.profile,
                                       sellingPrice = sellingPrice,
                                       description = description,
                                       condition = condition,
                                       textbook = textbook,
                                        )
            return self.form_valid(form)
        else:
            textbook_slug = kwargs['slug']
            textbook = Textbook.objects.get(slug = textbook_slug)
            sellingPosts = textbook.sellingpost_set.all
            context = kwargs
            context['textbook'] = textbook
            context['sellingPosts'] = sellingPosts
            context['form'] = form
            context['view'] = self
            return render(request,'book/textbook_detail.html', context)

    def form_valid(self, form):
        print(form.cleaned_data['sellingPrice'])
        return super(TextbookDetailView, self).form_valid(form)

    def get_success_url(self):
        return reverse_lazy('book:book-listing', kwargs={'slug': self.object.slug})

    def get_context_data(self, *args, **kwargs):
        slugger = kwargs.get('object').slug
        textbook = get_object_or_404(Textbook, slug = slugger)
        sellingPosts = textbook.sellingpost_set.all
        context = super(TextbookDetailView, self).get_context_data(*args, **kwargs)
        print(context)
        context['textbook'] = textbook
        context['sellingPosts'] = sellingPosts
        context['form'] = SellingPostCreateForm()
        return context
