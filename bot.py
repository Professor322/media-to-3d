import os
import shutil
from enum import IntEnum
from urllib.request import urlretrieve

import telebot
from pydantic import BaseModel

BOT_TOKEN = os.environ.get("BOT_TOKEN")
CUSTOMER_DATA_DIR = "./customer"

bot = telebot.TeleBot(BOT_TOKEN)


class InputType(IntEnum):
    VIDEO = (1,)
    ARHIVE_REAL_PHOTOS = (2,)
    ARHIVE_SYNTHETIC = (3,)
    UNKNOWN = (4,)


class RenderType(IntEnum):
    VIDEO = (1,)
    MESH = (2,)
    UNKNOWN = 3


class Config(BaseModel):
    data_type: InputType = InputType.ARHIVE_SYNTHETIC
    render_type: RenderType = RenderType.VIDEO


config = Config()


@bot.message_handler(commands=["start"])
def send_welcome(message):
    bot.send_message(
        message.chat.id,
        f"This bot is used to trasform photos and videos into 3D objects. As as a result of execution this bots returns:\n"
        f"({RenderType.VIDEO}) video\n({RenderType.MESH}) 3D mesh (not yet supported)",
        parse_mode="Markdown",
    )
    sent_msg = bot.send_message(
        message.chat.id,
        f"Please specify the type of data we will be using. Currently supported:\n"
        f"({InputType.VIDEO}) videos",
        parse_mode="Markdown",
    )
    bot.register_next_step_handler(sent_msg, parse_data_type)


def parse_data_type(message):
    sent_msg = None
    data_type = int(message.text)
    if data_type == InputType.ARHIVE_REAL_PHOTOS:
        config.data_type = InputType.ARHIVE_REAL_PHOTOS
    elif data_type == InputType.ARHIVE_SYNTHETIC:
        config.data_type = InputType.ARHIVE_SYNTHETIC
    elif data_type == InputType.VIDEO:
        config.data_type = InputType.VIDEO
    else:
        config.data_type = InputType.UNKNOWN
        bot.send_message(message.chat.id, "Incorrect data type, please restart")
        return

    sent_msg = bot.send_message(
        message.chat.id,
        f"Selected data type {config.data_type}\n"
        f"What type of render to return?\n"
        f"({RenderType.VIDEO}) video\n"
        f"({RenderType.MESH}) mesh (not supported)",
    )
    if sent_msg != None:
        bot.register_next_step_handler(sent_msg, parse_render_type)


def parse_render_type(message):
    sent_msg = None
    render_type = int(message.text)
    if render_type == RenderType.VIDEO:
        config.render_type = RenderType.VIDEO
    elif render_type == RenderType.MESH:
        config.render_type = RenderType.MESH
    else:
        config.data_type = RenderType.UNKNOWN
        bot.send_message(message.chat.id, "Incorrect render_type, please restart")
        return
    sent_msg = bot.send_message(
        message.chat.id,
        f"Selected render type {config.render_type}\n" f"Waiting for your data...",
    )
    if sent_msg != None:
        bot.register_next_step_handler(sent_msg, handle_input_data)


def download_file(file_id):
    local_file_path = CUSTOMER_DATA_DIR + "/input_video.mov"
    file = bot.get_file(file_id)
    url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file.file_path}"
    urlretrieve(url=url, filename=local_file_path)
    return local_file_path


def handle_archive(file):
    raise NotImplementedError("Not implemented")


def handle_video(video):
    download_file(video.file_id)


def process(chat_id):
    # preprocess video
    # start training
    # render video
    bot.send_message(chat_id, "Ready. Uploading result...")


def handle_input_data(message):
    if os.path.exists(CUSTOMER_DATA_DIR):
        print("Dir exists, clearing")
        shutil.rmtree(CUSTOMER_DATA_DIR)
    os.mkdir(CUSTOMER_DATA_DIR)

    expect_archive = config.data_type in [
        InputType.ARHIVE_REAL_PHOTOS,
        InputType.ARHIVE_SYNTHETIC,
    ]
    if expect_archive:
        file = message.document
        if file.mime_type == "application/zip":
            handle_archive(file)
        else:
            bot.send_message(message.chat.id, "incorrect datatype. restart")
            return
    else:
        video = message.video
        handle_video(video)
    bot.send_message(message.chat.id, "Data received. Processing might take some time...")
    process(message.chat.id)


bot.infinity_polling()
