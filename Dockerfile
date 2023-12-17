FROM python:3.10-slim
WORKDIR /app
COPY ./requirements.txt /app
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8080
ENV FLASK_APP=main.py
CMD ["flask", "run", "--host", "0.0.0.0"]