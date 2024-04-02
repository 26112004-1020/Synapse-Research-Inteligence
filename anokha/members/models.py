from django.contrib.auth.models import AbstractUser
from django.db import models
from phonenumber_field.modelfields import PhoneNumberField



class CustomUser(AbstractUser):

    phone_no = PhoneNumberField("Phone No.",null=True)

    def __str__(self):
        return self.username
