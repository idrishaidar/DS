from django import forms
from django.contrib.auth.forms import UserCreationForm
from .models import CustomUser, CHILD_AGE_CHOICES


class RegisterForm(UserCreationForm):
    email = forms.EmailField(
        widget=forms.EmailInput(attrs={'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-green-500', 'placeholder': 'your@email.com'})
    )
    full_name = forms.CharField(
        max_length=200,
        required=False,
        widget=forms.TextInput(attrs={'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-green-500', 'placeholder': 'Your full name'})
    )
    child_age_range = forms.ChoiceField(
        choices=[('', 'Select an option...')] + CHILD_AGE_CHOICES,
        required=False,
        widget=forms.Select(attrs={'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-green-500'})
    )
    password1 = forms.CharField(
        label='Password',
        widget=forms.PasswordInput(attrs={'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-green-500', 'placeholder': 'Password'})
    )
    password2 = forms.CharField(
        label='Confirm Password',
        widget=forms.PasswordInput(attrs={'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-green-500', 'placeholder': 'Confirm password'})
    )

    class Meta:
        model = CustomUser
        fields = ('email', 'full_name', 'child_age_range', 'password1', 'password2')


class ProfileForm(forms.ModelForm):
    class Meta:
        model = CustomUser
        fields = ('full_name', 'child_age_range', 'bio', 'avatar')
        widgets = {
            'full_name': forms.TextInput(attrs={'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-green-500'}),
            'child_age_range': forms.Select(attrs={'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-green-500'}),
            'bio': forms.Textarea(attrs={'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-green-500', 'rows': 4}),
        }
