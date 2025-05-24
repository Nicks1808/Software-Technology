# Χρησιμοποιούμε την επίσημη εικόνα Python 3.10
FROM python:3.10-slim

# Ορισμός του working directory για την εφαρμογή
WORKDIR /app

# Ορισμός της μεταβλητής περιβάλλοντος για να γνωρίζουμε ότι τρέχουμε σε Docker
ENV DOCKER=true

# Εγκατάσταση απαιτούμενων βιβλιοθηκών για την υποστήριξη του Streamlit και άλλων βιβλιοθηκών
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxml2 \
    libxslt1-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Αντιγραφή του requirements.txt στο Docker container και εγκατάσταση των απαιτούμενων βιβλιοθηκών
COPY requirements.txt . 
RUN pip install --upgrade pip 
RUN pip install -r requirements.txt

# Αντιγραφή του app.py στο container
COPY app.py .

# Εκθέτουμε την θύρα 8501 για την εφαρμογή Streamlit
EXPOSE 8501

# Εκκίνηση της εφαρμογής Streamlit στον Docker container
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
