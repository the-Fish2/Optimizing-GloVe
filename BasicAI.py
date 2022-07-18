greetings = ["Hello", "What's up?", "Howdy", "Greetings"]
goodbyes = ["Bye", "Goodbye", "See you later", "See you soon"]

keywords = ["music"]
responses = ["I love music too"]

import random
print(random.choice(greetings))

user = ""

while (user != "bye"):
    keyword_found = False
    user = input("Say something\n")
    user = user.lower()

    for index in range(len(keywords)):
        if (keywords[index] in user):
            print("Bot: " + responses[index])
            keyword_found = True

    if not keyword_found:
        new = input("What keyword should I respond to? \n")
        keywords.append(new)
        new_response = input("How should I respond to " + new + "? \n")
        responses.append(new_response)