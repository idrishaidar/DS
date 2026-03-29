# Montessori Academy

A Montessori-focused MOOC platform for parents, educators, and adult learners. Built with Django.

## Requirements

- Python 3.10+
- pip

## Quick Start

### 1. Clone the repo and navigate to the project

```bash
cd montessori_mooc
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements/dev.txt
```

### 4. Apply database migrations

```bash
python manage.py migrate
```

### 5. Create a superuser (for the admin panel)

```bash
python manage.py createsuperuser
```

### 6. Run the development server

```bash
python manage.py runserver
```

The app will be available at **http://127.0.0.1:8000**

---

## Key URLs

| URL | Description |
|---|---|
| `http://127.0.0.1:8000/` | Home page |
| `http://127.0.0.1:8000/courses/` | Course catalogue |
| `http://127.0.0.1:8000/dashboard/` | User dashboard |
| `http://127.0.0.1:8000/admin/` | Admin panel (manage courses) |
| `http://127.0.0.1:8000/accounts/register/` | Register |
| `http://127.0.0.1:8000/accounts/login/` | Login |

---

## Adding Your First Course

1. Go to `http://127.0.0.1:8000/admin/` and log in with your superuser account
2. Under **Courses**, click **Add Course** — fill in title, description, category, level, and set **Published** to true
3. Add **Modules** and **Lessons** inside the course (inline in the admin)
4. Visit `/courses/` to see the course appear in the catalogue

---

## Project Structure

```
montessori_mooc/
├── config/             # Django settings, urls, wsgi
│   └── settings/
│       ├── base.py     # Shared settings
│       └── dev.py      # Development settings (SQLite, debug toolbar)
├── accounts/           # Custom user model, auth views, dashboard
├── courses/            # Course, Module, Lesson, Enrollment, Progress
├── templates/          # All HTML templates (TailwindCSS CDN)
├── static/             # CSS and static assets
├── media/              # Uploaded images (avatars, course covers)
├── requirements/
│   ├── base.txt        # Production dependencies
│   └── dev.txt         # Dev dependencies (includes base)
└── manage.py
```

---

## Environment Variables

For production, set these in a `.env` file (never commit it):

```
SECRET_KEY=your-secret-key-here
DEBUG=False
ALLOWED_HOSTS=yourdomain.com
DATABASE_URL=postgres://user:password@host:5432/dbname
```

---

## Implementation Phases

- [x] Phase 1 — Foundation (models, views, templates, auth)
- [ ] Phase 2 — Learning experience (enrollment, progress tracking, dashboard)
- [ ] Phase 3 — Completion loop (quizzes, certificates)
- [ ] Phase 4 — Polish (search, email notifications, SEO)
