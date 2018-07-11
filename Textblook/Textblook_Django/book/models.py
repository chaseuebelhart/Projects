from django.db import models
from django.db.models.signals import pre_save, post_save
from .utils import unique_slug_generator

class Textbook(models.Model):
    title = models.CharField(max_length=200)
    author = models.CharField(max_length=100)
    isbn = models.CharField(max_length=15)
    image = models.CharField(max_length=200)
    klass = models.CharField(max_length = 20)
    timestamp = models.DateTimeField(auto_now_add = True)
    updated = models.DateTimeField(auto_now = True)
    slug = models.SlugField(unique = True, null = True, blank = True)

    def __str__(self):
        return self.title


def tb_pre_save_reciever(sender, instance, *args, **kwargs):
    if not instance.slug:
        instance.slug = unique_slug_generator(instance)

# def tb_post_save_reciever(sender, instance, *args, **kwargs):
#     print("hi")

pre_save.connect(tb_pre_save_reciever, sender = Textbook)
# post_save.connect(tb_post_save_reciever, sender = Textbook)
