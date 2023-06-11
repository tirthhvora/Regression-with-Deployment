FROM python:3.7
COPY . /app
WORKDIR /app
RUN pip install -r Requirements.txt
EXPOSE 5000
CMD ["python", "app.py"]


