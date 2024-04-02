
from django.urls import path
from. import views

from django.contrib.auth import views as auth_views

urlpatterns = [
    path('login/',views.login_user,name="login"),
    path('logout/',views.logout_user,name="logout"),
    path('reg_user/',views.reg_user,name="reg_user"),
    path('edit_profile/', views.edit_profile, name='edit_profile'), 
]