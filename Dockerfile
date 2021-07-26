FROM python:3.7-alpine
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
RUN mkdir /Eniac2021-TripleVAE-Events
WORKDIR /Eniac2021-TripleVAE-Events
RUN pip install --no-cache-dir --upgrade pip pipenv
RUN apk update && apk add --no-cache gcc bash
COPY Pipfile Pipfile.lock ./
RUN pipenv install --system --deploy --ignore-pipfile