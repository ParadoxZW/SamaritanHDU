# Generated by Django 2.2.3 on 2019-09-24 16:25

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('course', '0002_auto_20190924_1620'),
    ]

    operations = [
        migrations.RenameField(
            model_name='student_course',
            old_name='u_username',
            new_name='u_id',
        ),
    ]
