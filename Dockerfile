FROM xychelsea/anaconda3:latest

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir magenta tensorflow==2.9.3 pretty_midi flask flask_cors

ENV PORT=8080
EXPOSE 8080

CMD ["python", "main.py"]
