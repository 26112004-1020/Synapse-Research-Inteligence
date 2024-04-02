from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login,logout
from django.contrib import messages
from django.contrib.auth.forms import UserCreationForm,UserChangeForm
from .forms import reg_user_form,CustomUserChangeForm
from django.urls import reverse_lazy
from django.contrib.auth.views import PasswordChangeView
from .forms import CustomPasswordChangeForm


# from .forms import reg_user_form,EditProfileForm 

def login_user(request):
    if request.method == 'POST':
        username = request.POST["username"]
        password = request.POST["password"]
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('home')
        else:
            messages.success(request,("there was an error loging in,try again"))
            return redirect('login')

    else:   
        return render(request, 'login.html', {})
    
def logout_user(request):
    logout(request)
    # messages.success(request,("logged out successfully"))
    return redirect('home')

def reg_user(request):
    if request.method == 'POST':
        form = reg_user_form(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data['username']
            password = form.cleaned_data['password1']
            user = authenticate(request, username=username, password=password)
            login(request, user)
            messages.success(request,("registration successful"))
            return redirect('home')
    else:
        form = reg_user_form 
    return render(request,'reg_user.html',{'form':form})

def edit_profile(request):
    if request.method == 'POST':
        form = CustomUserChangeForm(request.POST, instance=request.user)
        if form.is_valid():
            form.save()
            messages.success(request, "Profile updated successfully")
            return redirect('profile')
    else:
        form = CustomUserChangeForm(instance=request.user)
    return render(request, 'edit_profile.html', {'form': form})




