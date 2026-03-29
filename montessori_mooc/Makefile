.PHONY: help install migrate superuser run shell reset lint

PYTHON  = python
MANAGE  = $(PYTHON) manage.py
SETTINGS = config.settings.dev

export DJANGO_SETTINGS_MODULE = $(SETTINGS)

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "  install     Create venv and install dev dependencies"
	@echo "  migrate     Apply database migrations"
	@echo "  superuser   Create a superuser account"
	@echo "  run         Start the development server"
	@echo "  shell       Open Django interactive shell"
	@echo "  reset       Drop the SQLite DB and re-migrate (dev only)"
	@echo "  lint        Run Django system checks"

install:
	$(PYTHON) -m venv venv
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -r requirements/dev.txt
	@echo ""
	@echo "Done. Activate the venv with: source venv/bin/activate"

migrate:
	$(MANAGE) migrate

superuser:
	$(MANAGE) createsuperuser

run:
	$(MANAGE) runserver

shell:
	$(MANAGE) shell

reset:
	rm -f db.sqlite3
	$(MANAGE) migrate

lint:
	$(MANAGE) check
