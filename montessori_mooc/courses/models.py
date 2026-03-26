from django.db import models
from django.utils.text import slugify
from accounts.models import CustomUser

LEVEL_CHOICES = [
    ('beginner', 'Beginner'),
    ('intermediate', 'Intermediate'),
    ('advanced', 'Advanced'),
]

CATEGORY_CHOICES = [
    ('philosophy', 'Montessori Philosophy'),
    ('environment', 'Prepared Environment'),
    ('activities', 'Practical Activities'),
    ('parenting', 'Parenting & Communication'),
    ('development', 'Child Development'),
]


class Course(models.Model):
    title = models.CharField(max_length=200)
    slug = models.SlugField(unique=True, blank=True)
    description = models.TextField()
    cover_image = models.ImageField(upload_to='courses/', blank=True, null=True)
    category = models.CharField(max_length=50, choices=CATEGORY_CHOICES)
    level = models.CharField(max_length=20, choices=LEVEL_CHOICES, default='beginner')
    instructor = models.ForeignKey(CustomUser, on_delete=models.SET_NULL, null=True, related_name='taught_courses')
    is_published = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = slugify(self.title)
        super().save(*args, **kwargs)

    def __str__(self):
        return self.title

    def get_absolute_url(self):
        from django.urls import reverse
        return reverse('courses:detail', kwargs={'slug': self.slug})


class Module(models.Model):
    course = models.ForeignKey(Course, on_delete=models.CASCADE, related_name='modules')
    title = models.CharField(max_length=200)
    order = models.PositiveIntegerField(default=0)

    class Meta:
        ordering = ['order']

    def __str__(self):
        return f"{self.course.title} - {self.title}"


class Lesson(models.Model):
    module = models.ForeignKey(Module, on_delete=models.CASCADE, related_name='lessons')
    title = models.CharField(max_length=200)
    video_url = models.URLField(blank=True)
    content = models.TextField(blank=True)
    duration_mins = models.PositiveIntegerField(default=0)
    order = models.PositiveIntegerField(default=0)

    class Meta:
        ordering = ['order']

    def __str__(self):
        return self.title


class Enrollment(models.Model):
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE, related_name='enrollments')
    course = models.ForeignKey(Course, on_delete=models.CASCADE, related_name='enrollments')
    enrolled_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        unique_together = ('user', 'course')

    def __str__(self):
        return f"{self.user.email} → {self.course.title}"


class Progress(models.Model):
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE, related_name='progress')
    lesson = models.ForeignKey(Lesson, on_delete=models.CASCADE, related_name='progress')
    completed_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('user', 'lesson')
