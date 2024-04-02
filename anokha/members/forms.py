from django.contrib.auth.forms import UserCreationForm,UserChangeForm
from django import forms
from phonenumber_field.formfields import PhoneNumberField
from django.contrib.auth.forms import PasswordChangeForm

from .models import CustomUser 

class reg_user_form(UserCreationForm):
    email = forms.EmailField()
    first_name = forms.CharField()
    last_name = forms.CharField()
    phone_no = PhoneNumberField()




    class Meta:
        model = CustomUser  # Use your CustomUser model here
        fields = ('first_name', 'last_name', 'email','phone_no','username', 'password1', 'password2')



    def save(self, commit=True):
        user = super().save(commit=False)
        user.first_name = self.cleaned_data['first_name']
        user.last_name = self.cleaned_data['last_name']

        if commit:
            user.save()

        # Create or update myuser instance
        

        return user
    


class CustomUserChangeForm(UserChangeForm):
    class Meta:
        model = CustomUser
        fields = ('first_name', 'last_name', 'email', 'phone_no')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['password'].widget = forms.HiddenInput()

    def get_form(self, *args, **kwargs):
        form = super().get_form(*args, **kwargs)
        form.fields.pop('password')
        return form  



class CustomPasswordChangeForm(PasswordChangeForm):
    class Meta:
        model = CustomUser  # Replace with your user model
        fields = '__all__'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['new_password1'].widget = forms.PasswordInput(attrs={'class': 'form-control'})
        self.fields['new_password2'].widget = forms.PasswordInput(attrs={'class': 'form-control'})