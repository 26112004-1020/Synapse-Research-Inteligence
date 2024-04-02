from django.shortcuts import render,redirect,get_object_or_404


from django.contrib import messages
from django.contrib.auth.decorators import login_required





from django.http import HttpResponseRedirect
from django.contrib.auth import authenticate, login,logout
from django.contrib import messages

from django.utils import timezone

def home(request):
    
    return render(request, 'home.html', {})
    
