from django.contrib import admin
from .models import Course, Module, Lesson, Enrollment, Progress


class LessonInline(admin.TabularInline):
    model = Lesson
    extra = 1
    fields = ['title', 'video_url', 'duration_mins', 'order']


class ModuleInline(admin.StackedInline):
    model = Module
    extra = 1
    fields = ['title', 'order']
    show_change_link = True


@admin.register(Course)
class CourseAdmin(admin.ModelAdmin):
    list_display = ['title', 'category', 'level', 'instructor', 'is_published', 'created_at']
    list_filter = ['category', 'level', 'is_published']
    search_fields = ['title', 'description']
    prepopulated_fields = {'slug': ('title',)}
    inlines = [ModuleInline]


@admin.register(Module)
class ModuleAdmin(admin.ModelAdmin):
    list_display = ['title', 'course', 'order']
    list_filter = ['course']
    inlines = [LessonInline]


@admin.register(Lesson)
class LessonAdmin(admin.ModelAdmin):
    list_display = ['title', 'module', 'duration_mins', 'order']
    list_filter = ['module__course']
    search_fields = ['title']


@admin.register(Enrollment)
class EnrollmentAdmin(admin.ModelAdmin):
    list_display = ['user', 'course', 'enrolled_at', 'completed_at']
    list_filter = ['course']
    search_fields = ['user__email', 'course__title']


@admin.register(Progress)
class ProgressAdmin(admin.ModelAdmin):
    list_display = ['user', 'lesson', 'completed_at']
    list_filter = ['lesson__module__course']
    search_fields = ['user__email']
