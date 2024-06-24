from chatterbot import ChatBot
bot = ChatBot(
    'Buddy',  
    logic_adapters=[
        'chatterbot.logic.BestMatch',
        'chatterbot.logic.TimeLogicAdapter'],
)
from chatterbot.trainers import ListTrainer

trainer = ListTrainer(bot)

trainer.train([
'Hi',
'Hello',
'I need your assistance regarding my health',
'Please, Provide me with your symptoms',
'I have a complaint.',
'Please elaborate, your concern',
'How long it will take to cure the illness ?',
'An order takes 3-5 days to get recovered.',
'Okay Thanks',
'No Problem! Have a Good Day!'
])


name=input("Enter Your Name: ")
print("Welcome to the Bot Service! Let me know how can I help you?")
while True:
    request=input(name+':')
    if request=='Bye' or request =='bye':
        print('Bot: Bye')
        break
    else:
        response=bot.get_response(request)
        print('Bot:',response)