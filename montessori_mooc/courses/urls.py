from django.urls import path
from . import views

app_name = 'courses'

urlpatterns = [
    path('', views.CourseListView.as_view(), name='list'),
    path('dashboard/', views.DashboardView.as_view(), name='dashboard'),
    path('<slug:slug>/', views.CourseDetailView.as_view(), name='detail'),
    path('<slug:slug>/enroll/', views.EnrollView.as_view(), name='enroll'),
    path('<slug:slug>/learn/', views.LessonFirstView.as_view(), name='lesson_first'),
    path('<slug:slug>/learn/<int:lesson_id>/', views.LessonView.as_view(), name='lesson'),
]
