# -*- coding: utf-8 -*-
# Generated by Django 1.11.3 on 2017-08-01 23:40
from __future__ import unicode_literals

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('book', '0007_textbook_sellingpost'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='textbook',
            name='sellingPost',
        ),
    ]
