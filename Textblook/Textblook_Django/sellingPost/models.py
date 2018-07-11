from django.db import models
from book.models import Textbook
from fundamentals.models import Profile
# Create your models here.
class SellingPost(models.Model):
    sellingPrice = models.DecimalField(max_digits=6, decimal_places =2)
    description = models.TextField()
    profile = models.ForeignKey(Profile, on_delete=models.CASCADE)
    textbook = models.ForeignKey(Textbook, on_delete=models.CASCADE, blank = True, null = True)
    timestamp = models.DateTimeField(auto_now_add = True, blank = True)
    updated = models.DateTimeField(auto_now = True, blank = True)

    USED_MARKINGS = 'Used with markings'
    USED_NO_MARKINGS = 'Used with minimal/no markings'
    NEW = 'New'
    CONDITION_CHOICES = (
        (USED_MARKINGS, 'Used with markings'),
        (USED_NO_MARKINGS, 'Used with minimal/no markings'),
        (NEW, 'New'),
    )
    condition = models.CharField(
        max_length=31,
        choices=CONDITION_CHOICES,
        default=NEW,
    )

    def __str__ (self):
        return str(self.id)

# def tb_pre_save_reciever(sender, instance, *args, **kwargs):
#     instance.profile = instance.user.profile
#     instance.textbook = instance.textbook
#
# pre_save.connect(tb_pre_save_reciever, sender = SellingPost)
