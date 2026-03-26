from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate
from django.contrib.auth.decorators import login_required
from django.utils.decorators import method_decorator
from django.views import View
from .forms import RegisterForm, ProfileForm


class RegisterView(View):
    def get(self, request):
        if request.user.is_authenticated:
            return redirect('courses:dashboard')
        form = RegisterForm()
        return render(request, 'accounts/register.html', {'form': form})

    def post(self, request):
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('courses:dashboard')
        return render(request, 'accounts/register.html', {'form': form})


@method_decorator(login_required, name='dispatch')
class ProfileView(View):
    def get(self, request):
        form = ProfileForm(instance=request.user)
        return render(request, 'accounts/profile.html', {'form': form})

    def post(self, request):
        form = ProfileForm(request.POST, request.FILES, instance=request.user)
        if form.is_valid():
            form.save()
            return redirect('accounts:profile')
        return render(request, 'accounts/profile.html', {'form': form})


class DashboardView(View):
    def get(self, request):
        return redirect('courses:dashboard')
