FROM python:3.12-slim

WORKDIR /media_to_3d


# Copy the requirements file into the container
COPY ./scripts /media_to_3d/scripts
COPY ./src /media_to_3d/src
COPY bot.py /media_to_3d/
COPY train.py /media_to_3d/
COPY render.py /media_to_3d/
COPY requirements.txt /media_to_3d/

# Install git
RUN apt update && apt install -y git && rm -rf /var/lib/apt/lists/*
RUN apt update && apt install -y libjpeg-dev zlib1g-dev && rm -rf /var/lib/apt/lists/*
RUN apt update && apt install -y ffmpeg && rm -rf /var/lib/apt/lists/*
RUN apt update && apt install -y gcc && rm -rf /var/lib/apt/lists/*
# Clone the repository
RUN git clone https://github.com/rmbrualla/pycolmap.git /media_to_3d/src/pycolmap


RUN pip install --upgrade pip
# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt


# Set a default value for the bot token (optional)
ENV TELEGRAM_BOT_TOKEN="default_token"

CMD ["python3", "bot.py"]
