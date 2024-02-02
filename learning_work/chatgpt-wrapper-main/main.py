# -*- coding: utf-8 -*-
from chatgpt_wrapper import ChatGPT

bot = ChatGPT()
print("start")
response = bot.ask("Hello, world!")
print("end")
print(response)  # prints the response from chatGPT