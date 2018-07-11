from django.db import models
from django.contrib.auth.models import User
from django.dispatch import receiver
from django.db.models.signals import pre_save, post_save
from fundamentals.utils import unique_slug_generator

class Profile(models.Model):
    user = models.OneToOneField(User, on_delete = models.CASCADE)
    facebook_username = models.CharField(max_length = 60, blank = True)
    university_email = models.EmailField(max_length = 30, blank = True)
    slug = models.SlugField(unique = True, null = True, blank = True)

    def __str__(self):
        return self.facebook_username

def tb_pre_save_reciever(sender, instance, *args, **kwargs):
    if not instance.slug:
        instance.slug = unique_slug_generator(instance)

pre_save.connect(tb_pre_save_reciever, sender = Profile)
