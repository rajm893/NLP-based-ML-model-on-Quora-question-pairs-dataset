FROM python:3.8
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8212
ENTRYPOINT ["uvicorn", "app:app", "--reload", "--host", "0.0.0.0", "--port", "8212"]