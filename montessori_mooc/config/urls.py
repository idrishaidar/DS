from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from courses.views import HomeView, DashboardView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', HomeView.as_view(), name='home'),
    path('dashboard/', DashboardView.as_view(), name='dashboard'),
    path('accounts/', include('accounts.urls')),
    path('courses/', include('courses.urls')),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
