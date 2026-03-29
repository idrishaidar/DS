from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.utils.decorators import method_decorator
from django.views import View
from django.db.models import Count, Q
from .models import Course, Lesson, Enrollment, Progress, CATEGORY_CHOICES, LEVEL_CHOICES


class HomeView(View):
    def get(self, request):
        featured_courses = Course.objects.filter(is_published=True).order_by('-created_at')[:6]
        return render(request, 'home.html', {'featured_courses': featured_courses})


class CourseListView(View):
    def get(self, request):
        courses = Course.objects.filter(is_published=True).select_related('instructor')
        category = request.GET.get('category', '')
        level = request.GET.get('level', '')
        if category:
            courses = courses.filter(category=category)
        if level:
            courses = courses.filter(level=level)
        enrolled_course_ids = set()
        if request.user.is_authenticated:
            enrolled_course_ids = set(
                Enrollment.objects.filter(user=request.user).values_list('course_id', flat=True)
            )
        return render(request, 'courses/list.html', {
            'courses': courses,
            'category_choices': CATEGORY_CHOICES,
            'level_choices': LEVEL_CHOICES,
            'selected_category': category,
            'selected_level': level,
            'enrolled_course_ids': enrolled_course_ids,
        })


class CourseDetailView(View):
    def get(self, request, slug):
        course = get_object_or_404(Course, slug=slug, is_published=True)
        modules = course.modules.prefetch_related('lessons').all()
        is_enrolled = False
        progress_lesson_ids = set()
        total_lessons = 0
        completed_lessons = 0

        if request.user.is_authenticated:
            is_enrolled = Enrollment.objects.filter(user=request.user, course=course).exists()
            if is_enrolled:
                all_lesson_ids = []
                for module in modules:
                    for lesson in module.lessons.all():
                        all_lesson_ids.append(lesson.id)
                total_lessons = len(all_lesson_ids)
                progress_lesson_ids = set(
                    Progress.objects.filter(
                        user=request.user,
                        lesson_id__in=all_lesson_ids
                    ).values_list('lesson_id', flat=True)
                )
                completed_lessons = len(progress_lesson_ids)

        progress_pct = int((completed_lessons / total_lessons * 100) if total_lessons > 0 else 0)

        return render(request, 'courses/detail.html', {
            'course': course,
            'modules': modules,
            'is_enrolled': is_enrolled,
            'progress_pct': progress_pct,
            'completed_lessons': completed_lessons,
            'total_lessons': total_lessons,
            'progress_lesson_ids': progress_lesson_ids,
        })


@method_decorator(login_required, name='dispatch')
class EnrollView(View):
    def post(self, request, slug):
        course = get_object_or_404(Course, slug=slug, is_published=True)
        Enrollment.objects.get_or_create(user=request.user, course=course)
        return redirect('courses:lesson_first', slug=slug)

    def get(self, request, slug):
        return redirect('courses:detail', slug=slug)


@method_decorator(login_required, name='dispatch')
class LessonView(View):
    def _get_lesson_and_check_enrollment(self, request, slug, lesson_id):
        course = get_object_or_404(Course, slug=slug, is_published=True)
        lesson = get_object_or_404(Lesson, id=lesson_id, module__course=course)
        enrollment = Enrollment.objects.filter(user=request.user, course=course).first()
        return course, lesson, enrollment

    def _get_nav_lessons(self, course, current_lesson):
        all_lessons = []
        for module in course.modules.prefetch_related('lessons').all():
            for lesson in module.lessons.all():
                all_lessons.append(lesson)
        current_idx = next((i for i, l in enumerate(all_lessons) if l.id == current_lesson.id), None)
        prev_lesson = all_lessons[current_idx - 1] if current_idx and current_idx > 0 else None
        next_lesson = all_lessons[current_idx + 1] if current_idx is not None and current_idx < len(all_lessons) - 1 else None
        return prev_lesson, next_lesson

    def _get_youtube_embed(self, video_url):
        if not video_url:
            return None
        import re
        patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/)([A-Za-z0-9_-]{11})',
        ]
        for pattern in patterns:
            match = re.search(pattern, video_url)
            if match:
                return f"https://www.youtube.com/embed/{match.group(1)}"
        return video_url

    def get(self, request, slug, lesson_id):
        course, lesson, enrollment = self._get_lesson_and_check_enrollment(request, slug, lesson_id)
        if not enrollment:
            return redirect('courses:detail', slug=slug)
        is_completed = Progress.objects.filter(user=request.user, lesson=lesson).exists()
        prev_lesson, next_lesson = self._get_nav_lessons(course, lesson)
        embed_url = self._get_youtube_embed(lesson.video_url)
        return render(request, 'courses/lesson.html', {
            'course': course,
            'lesson': lesson,
            'is_completed': is_completed,
            'prev_lesson': prev_lesson,
            'next_lesson': next_lesson,
            'embed_url': embed_url,
        })

    def post(self, request, slug, lesson_id):
        course, lesson, enrollment = self._get_lesson_and_check_enrollment(request, slug, lesson_id)
        if not enrollment:
            return redirect('courses:detail', slug=slug)
        Progress.objects.get_or_create(user=request.user, lesson=lesson)

        # Mark enrollment complete if all lessons are done
        if not enrollment.completed_at:
            all_lesson_ids = [
                l.id
                for module in course.modules.prefetch_related('lessons').all()
                for l in module.lessons.all()
            ]
            completed_count = Progress.objects.filter(
                user=request.user, lesson_id__in=all_lesson_ids
            ).count()
            if completed_count == len(all_lesson_ids) and all_lesson_ids:
                from django.utils import timezone
                enrollment.completed_at = timezone.now()
                enrollment.save()

        return redirect('courses:lesson', slug=slug, lesson_id=lesson_id)


@method_decorator(login_required, name='dispatch')
class LessonFirstView(View):
    """Redirect to the first incomplete lesson, or the first lesson if none started."""
    def get(self, request, slug):
        course = get_object_or_404(Course, slug=slug, is_published=True)
        enrollment = Enrollment.objects.filter(user=request.user, course=course).first()
        if not enrollment:
            return redirect('courses:detail', slug=slug)

        all_lessons = [
            lesson
            for module in course.modules.prefetch_related('lessons').all()
            for lesson in module.lessons.all()
        ]
        if not all_lessons:
            return redirect('courses:detail', slug=slug)

        completed_ids = set(
            Progress.objects.filter(
                user=request.user,
                lesson_id__in=[l.id for l in all_lessons]
            ).values_list('lesson_id', flat=True)
        )
        # Resume at first incomplete lesson; fall back to first lesson if all done
        next_lesson = next((l for l in all_lessons if l.id not in completed_ids), all_lessons[0])
        return redirect('courses:lesson', slug=slug, lesson_id=next_lesson.id)


@method_decorator(login_required, name='dispatch')
class DashboardView(View):
    def get(self, request):
        enrollments = Enrollment.objects.filter(user=request.user).select_related('course').prefetch_related(
            'course__modules__lessons'
        )
        enrollment_data = []
        for enrollment in enrollments:
            course = enrollment.course
            all_lesson_ids = []
            for module in course.modules.all():
                for lesson in module.lessons.all():
                    all_lesson_ids.append(lesson.id)
            total = len(all_lesson_ids)
            completed = Progress.objects.filter(
                user=request.user,
                lesson_id__in=all_lesson_ids
            ).count() if all_lesson_ids else 0
            pct = int((completed / total * 100) if total > 0 else 0)

            # Find next lesson to continue
            completed_ids = set(
                Progress.objects.filter(user=request.user, lesson_id__in=all_lesson_ids).values_list('lesson_id', flat=True)
            )
            next_lesson = None
            for lesson_id in all_lesson_ids:
                if lesson_id not in completed_ids:
                    next_lesson = lesson_id
                    break

            enrollment_data.append({
                'enrollment': enrollment,
                'course': course,
                'total': total,
                'completed': completed,
                'pct': pct,
                'next_lesson_id': next_lesson,
            })

        return render(request, 'accounts/dashboard.html', {
            'enrollment_data': enrollment_data,
        })
