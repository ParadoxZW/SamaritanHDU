# Generated by Django 2.2.3 on 2019-09-22 13:04

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='Course',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('c_id', models.CharField(max_length=10)),
                ('c_name', models.CharField(max_length=30)),
                ('c_student', models.ManyToManyField(related_name='stu_course_st', to=settings.AUTH_USER_MODEL)),
                ('c_teacher', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='tea_course_st', to=settings.AUTH_USER_MODEL)),
            ],
        ),
    ]
