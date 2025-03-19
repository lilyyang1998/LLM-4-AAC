import gradio as gr
from openai import OpenAI
import json
import logging
import os
from transformers import pipeline
import numpy as np
from conversational_agent import *
import random

DEBUG = False

js = """
function record() {
    let dialogueButton = document.getElementById(`dialogue_typein_btn`);

    // Assign a click event listener to the button
    if (dialogueButton) {
        dialogueButton.addEventListener('click', function() {
            // When the button is clicked, find the target button
            var targetButton = document.querySelector('.record.record-button.svelte-k4irr3');
                
            // Click the target button
            if (targetButton) {
                targetButton.click();
            }
        });
    }
    
    for (let i = 0; i <= 5; i++) {
    // Freq Used Words
    let freqButton = document.getElementById(`freq_${i}`);
        if (freqButton) {
            freqButton.addEventListener('click', function() {
                var targetButton = document.querySelector('.record.record-button.svelte-k4irr3');
                
                if (targetButton) {
                    targetButton.click();
                }
            });
        }
    }
}
"""


# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

client = OpenAI(
    api_key = os.environ.get("OPENAI_API_KEY")
)

llava_model, llava_processor = init_llava(DEBUG)


def switch2role(name, gender, age):
    print(f"User Persona: {name}, Gender: {gender}, Age: {age}")
    return {
        user_persona: gr.Column(visible=False),
        acquaintance_level: gr.Column(visible=True)
    }


def switch2upload(level):
    print(f"Level of Acquaintnace: {level}/10")
    return {
        acquaintance_level: gr.Column(visible=False),
        upload_file: gr.Column(visible=True)
    }


def switch2chat(image, file):

    return {
        upload_file: gr.Column(visible=False),
        chat_page: gr.Column(visible=True)
    }


def update_topics(pronouns, age, partner, detail, level, context):

    pronoun, poss_pronoun = pronouns[0].split('/')
    if pronoun.lower() == 'she':
        gender = 'female'
    elif pronoun.lower() == 'he':
        gender= 'male'
    else:
        gender = 'person'

    prompt_with_conv = """
    Keywords for a dialog are a group of words that can be used to write the complete dialog. Writing the dialog from the keywords may or may not need some additional context. For example,

    KEYWORDS: hey going
    DIALOG: Hey, how's it going?

    KEYWORDS: hi name
    CONTEXT: My name is Sheila, I am a 32-year-old female. My pronouns are she/her.
    DIALOG: Hi, my name is Sheila.

    In this exercise, we will generate keywords to write dialogs that can be said by a particular user at the beginning of a conversation. The user is {name}, a {age} year old {gender}. {name} is in a conversation with {poss_pronoun} {partner_role}. {name} says this about the {partner_role} {pronoun} is talking to: "{detail}". This is the conversation they have so far:
    CONVERSATION:
    {conversation}
    
    Generate six KEYWORDS that can be used to write a dialog that {name} can say next in this conversation with {poss_pronoun} {partner_role}.
    Generate your answer in the form of a json list. Start and end your answer with square brackets.
    """

    prompt_with_conv_context = """
    Keywords for a dialog are a group of words that can be used to write the complete dialog. Writing the dialog from the keywords may or may not need some additional context. For example,

    KEYWORDS: hey going
    DIALOG: Hey, how's it going?

    KEYWORDS: hi name
    CONTEXT: My name is Sheila, I am a 32-year-old female. My pronouns are she/her.
    DIALOG: Hi, my name is Sheila.

    In this exercise, we will generate keywords to write dialogs that can be said by a particular user at the beginning of a conversation. The user is {name}, a {age} year old {gender}. {name} is in a conversation with {poss_pronoun} {partner_role}. {name} says this about the {partner_role} {pronoun} is talking to: "{detail}". {name} wants to talk about the following things in the conversation:
    CONTEXT:
    {context}
    
    This is the conversation they have so far:
    CONVERSATION:
    {conversation}
    
    Generate six KEYWORDS that can be used to write a dialog that {name} can say next in this conversation with {poss_pronoun} {partner_role} based on the context. Generate your answer in the form of a json list. Start and end your answer with square brackets.
    """


    if len(conversation) != 0:

        if context is None or len(context) == 0:

            conversation_string = "\n"
            for d in conversation:
                conversation_string += d["speaker"] + ' says ,' + d["text"] + "\n"

            output_list = get_chatgpt_output(prompt_with_conv.format(name=name, age=age, gender=gender, pronoun=pronoun, poss_pronoun=poss_pronoun, partner_role=partner, detail=detail, conversation=conversation_string))
            print(f'continue keywords: {output_list}')

            return output_list
        
        else:

            conversation_string = "\n"
            for d in conversation:
                conversation_string += d["speaker"] + ' says ,' + d["text"] + "\n"

            output_list_1 = get_chatgpt_output(prompt_with_conv.format(name=name, age=age, gender=gender, pronoun=pronoun, poss_pronoun=poss_pronoun, partner_role=partner, detail=detail, conversation=conversation_string))
            print(f'continue keywords (without context): {output_list_1}')

            output_list_2 = get_chatgpt_output(prompt_with_conv_context.format(name=name, age=age, gender=gender, pronoun=pronoun, poss_pronoun=poss_pronoun, partner_role=partner, detail=detail, conversation=conversation_string, context=context))
            print(f'continue keywords (with context): {output_list_2}')

            output_list = random.sample(output_list_1, 3) + random.sample(output_list_2, 3)
            print(f'continue keywords: {output_list}')

            return output_list

    else:

        prompt = """
        Keywords for a dialog are a group of words that can be used to write the complete dialog. Writing the dialog from the keywords may or may not need some additional context. For example,

        KEYWORDS: hey going
        DIALOG: Hey, how's it going?

        KEYWORDS: hi name
        CONTEXT: My name is Sheila, I am a 32-year-old female. My pronouns are she/her.
        DIALOG: Hi, my name is Sheila.

        In this exercise, we will generate keywords to write dialogs that can be said by a particular user at the beginning of a conversation. The user is {name}, a {age} year old {gender}. {name} is beginning a conversation with {poss_pronoun} {partner_role}. {name} says this about the {partner_role} {pronoun} is talking to: "{detail}". Generate six KEYWORDS that can be used to write the first thing that {name} says in this conversation with {poss_pronoun} {partner_role}.

        Generate your answer in the form of a json list. Start and end your answer with square brackets.
        """
        output_list = get_chatgpt_output(prompt.format(name=name, age=age, gender=gender, pronoun=pronoun, poss_pronoun=poss_pronoun, partner_role=partner, detail=detail))
        print(f'begin keywords: {output_list}')

        return output_list


conversation = []
curr_turn = {}

def generate_dialogues(keyword, context, pronouns, age, partner, detail, level):
    if 'image_caption' in curr_turn:
        message = generate_dialogue_prompt_using_image(keyword, curr_turn['image_caption'])
    elif keyword in [None, '']:
        message = generate_dialogue_no_keyword()
    else:
        message = generate_dialogue_prompt(keyword, context, pronouns, age, partner, detail, level)
    output = get_chatgpt_output(message)
    print(f'keyword: {keyword}')
    print(f'dialogues: {output}')
    logger.info("Get %s dialogues from LLM" % len(output))
    return output

def generate_dialogue_prompt_using_image(topic, image_caption):

    if len(conversation) == 0: 
        prompt = ("Write ten diverse sentences that a USER might say at the beginning of a conversation based on a given keyword and the caption of an image shared by the USER."
                  "Examples:"
                  "INPUT: keywords - movie, emotion - neutral, acquaintance - 9/10"
                  "IMAGE CAPTION: In this image, three people are in a snowy landscape, one kicking another, who is falling, while the third person watches."
                  "OUTPUT: Hey! Let's watch an action movie tonight! I am craving some asskicking and fight scenes."
                  "INPUT: keywords - shopping, emotion - happy, acquaintance - 7/10"
                  "IMAGE CAPTION: Woman in blue lace dress with hands on hips."
                  "OUTPUT: Omg, I went shopping today and found a nice blue dress for myself."
                  "INPUT: keywords - pain hand swell, emotion - neutral, acquaintance - 5/10"
                  "IMAGE CAPTION: In the image, a pair of hands are shown with a bandage on the thumb of the left hand. The hands seem to swollen slightly."
                  "OUTPUT: Hello. My hands are swollen and it is painful."
                  "The sentences should be written in the present tense, and can either be a question or a comment."
                  "Please also take the acquaintance level and emotion of the user into account for generating sentences."
                  "(If the level is higher, the dialogues are more casual, intimate and open because they are closer.)"
                  "(On the other hand, if the level is lower, the dialogues should maintain a respectful distance and share less personal information.)"
                  "In the conversation, there are two speakers: the USER and the SPEAKER."
                  "USER PERSONA: pronouns - {}, age - {}; PARTNER RELATIONSHIP: {}, {}"
                  "INPUT: keywords - {}, emotion - {}, acquaintance - {}/10"
                  "IMAGE CAPTION: {}"
                  "Generate the list of ten sentences in the form of a json list. Each sentence is a string within 10 words."
                  "Start and end your output with a square bracket."
                  "OUTPUT: ")
        return prompt.format(pronouns, age, partner, detail, topic, emotion, level, image_caption)
    
    else:
        conversation_string = ""
        for d in conversation:
            conversation_string += d["speaker"] + ' says ,' + d["text"] + "\n"

        prompt = ("Write ten diverse sentences that a USER might say next in the conversation based on a given keyword and the caption of an image shared by the USER."
                  "Examples:"
                  "INPUT: keywords - movie, emotion - neutral, acquaintance - 9/10"
                  "IMAGE CAPTION: In this image, three people are in a snowy landscape, one kicking another, who is falling, while the third person watches."
                  "OUTPUT: Let's watch an action movie! I am craving some asskicking and fight scenes."
                  "INPUT: keywords - shopping, emotion - happy, acquaintance - 7/10"
                  "IMAGE CAPTION: Woman in blue lace dress with hands on hips."
                  "OUTPUT: I went shopping today! Found a nice blue dress for myself."
                  "INPUT: keywords - pain hand swell, emotion - neutral, acquaintance - 5/10"
                  "IMAGE CAPTION: In the image, a pair of hands are shown with a bandage on the thumb of the left hand. The hands seem to swollen slightly."
                  "OUTPUT: Look, my hands are swollen and it is painful."
                  "The sentences should be written in the present tense, and can either be a question or a comment."
                  "Please also take the acquaintance level and emotion of the user into account for generating sentences."
                  "(If the level is higher, the dialogues are more casual, intimate and open because they are closer.)"
                  "(On the other hand, if the level is lower, the dialogues should maintain a respectful distance and share less personal information.)"
                  "In the conversation, there are two speakers: the USER and the SPEAKER."
                  "USER PERSONA: pronouns - {}, age - {}; PARTNER RELATIONSHIP: {}, {}"
                  "CONVERSATION: \n{}\n"
                  "INPUT: keywords - {}, emotion - {}, acquaintance - {}/10"
                  "IMAGE CAPTION: {}"
                  "Generate the list of ten sentences that USER might say next in the conversation in the form of a json list. Each sentence is a string."
                  "Start and end your output with a square bracket."
                  "OUTPUT: ")

        return prompt.format(pronouns, age, partner, detail, conversation, topic, emotion, level, image_caption)
    
def generate_dialogue_no_keyword():
    if len(conversation) == 0: 
        prompt = ("Write ten diverse sentences that a USER might say at the beginning of a conversation based on their persona."
                  "The sentences should be written in the present tense, and can either be a question or a comment."
                  "Please also take the acquaintance level and emotion of the user into account for generating sentences."
                  "(If the level is higher, the dialogues are more casual, intimate and open because they are closer.)"
                  "(On the other hand, if the level is lower, the dialogues should maintain a respectful distance and share less personal information.)"
                  "In the conversation, there are two speakers: the USER and the SPEAKER."
                  "USER PERSONA: pronouns - {}, age - {}; PARTNER RELATIONSHIP: {}, {}"
                  "INPUT: emotion - {}, acquaintance - {}/10"
                  "Generate the list of ten sentences in the form of a json list. Each sentence is a string."
                  "Start and end your output with a square bracket."
                  "OUTPUT: ")
        return prompt.format(pronouns, age, partner, detail, emotion, level)
    else:
        conversation_string = ""
        for d in conversation:
            conversation_string += d["speaker"] + ' says ,' + d["text"] + "\n"

        prompt = ("Write ten diverse sentences that a USER might say next in the conversation."
                  "The sentences should be written in the present tense, and can either be a question or a comment."
                  "Please also take the acquaintance level and emotion of the user into account for generating sentences."
                  "(If the level is higher, the dialogues are more casual, intimate and open because they are closer.)"
                  "(On the other hand, if the level is lower, the dialogues should maintain a respectful distance and share less personal information.)"
                  "In the conversation, there are two speakers: the USER and the SPEAKER."
                  "USER PERSONA: pronouns - {}, age - {}; PARTNER RELATIONSHIP: {}, {}"
                  "CONVERSATION: \n{}\n"
                  "INPUT: emotion - {}, acquaintance - {}/10"
                  "Generate the list of ten sentences that USER might say next in the conversation in the form of a json list. Each sentence is a string."
                  "Start and end your output with a square bracket."
                  "OUTPUT: ")
        
        return prompt.format(pronouns, age, partner, detail, conversation, emotion, level)

def generate_dialogue_prompt(keywords, context, pronouns, age, partner, detail, level):

    pronoun, poss_pronoun = pronouns[0].split('/')
    if pronoun.lower() == 'she':
        gender = 'female'
    elif pronoun.lower() == 'he':
        gender= 'male'
    else:
        gender = 'person'

    conversation_string = ""
    for d in conversation:
        conversation_string += d["speaker"] + ' said ,' + d["text"] + "\n"


    prompt_no_conv = """
    Keywords for a dialog are a group of words that can be used to write the complete dialog. For example,

    KEYWORDS: okay
    DIALOG: It's going okay

    KEYWORDS: page turn
    DIALOG: Can you turn pages for me?

    In this exercise, we will generate keywords to write dialogs that can be said by a particular user in a conversation. The user is {name}, a {age} year old {gender}. {name} is beginning a conversation with {poss_pronoun} {partner_role}. {name} says this about the {partner_role} {pronoun} is talking to: "{detail}". Write six diverse dialogs that {name} can say to begin the conversation with {poss_pronoun} {partner_role} based on the following keywords. Please also take into account that {name} has an acquaintance of {level} on a scale of 10 with {partner_role} and is currently feeling {emotion}. Generate your answer in the form of a json list where each entry is a string. Start and end your answer with square brackets.

    KEYWORDS: {keywords}
    DIALOG: 
    """


    prompt_with_conv = """
    Keywords for a dialog are a group of words that can be used to write the complete dialog. For example,

    KEYWORDS: okay
    DIALOG: It's going okay

    KEYWORDS: page turn
    DIALOG: Can you turn pages for me?

    Sometimes, the conversation that has happened so far can be used to write the dialog from keywords. For example,

    CONVERSATION: 
    Brian said, "Hello, how are you doing?"
    Chet said, "Pretty good, thanks. And yourself?"
    Brian said, "Awesome, I just got back from a bike ride."

    Write the next thing that Chet can say in the conversation based on the following keywords.
    KEYWORDS: time bike
    DIALOG: Cool, do you spend a lot of time biking?

    In this exercise, we will generate keywords to write dialogs that can be said by a particular user in a conversation. The user is {name}, a {age} year old {gender}. {name} is in a conversation with {poss_pronoun} {partner_role}. {name} says this about the {partner_role} {pronoun} is talking to: "{detail}".  This is the conversation they have so far:
    CONVERSATION:
    {conversation}

    Write six diverse dialogs that {name} can say next in the conversation with {poss_pronoun} {partner_role} based on the following keywords. Please also take into account that {name} has an acquaintance of {level} on a scale of 10 with {partner_role} and is currently feeling {emotion}. Generate your answer in the form of a json list where each entry is a string. Start and end your answer with square brackets.

    KEYWORDS: {keywords}
    DIALOG: 
    """

    prompt_with_conv_and_context = """
    Keywords for a dialog are a group of words that can be used to write the complete dialog. For example,

    KEYWORDS: okay
    DIALOG: It's going okay

    KEYWORDS: page turn
    DIALOG: Can you turn pages for me?

    Sometimes, the conversation that has happened so far can be used to write the dialog from keywords. For example,

    CONVERSATION: 
    Brian said, "Hello, how are you doing?"
    Chet said, "Pretty good, thanks. And yourself?"
    Brian said, "Awesome, I just got back from a bike ride."

    Write the next thing that Chet can say in the conversation based on the following keywords.
    KEYWORDS: time bike
    DIALOG: Cool, do you spend a lot of time biking?

    In this exercise, we will generate keywords to write dialogs that can be said by a particular user in a conversation. The user is {name}, a {age} year old {gender}. {name} is in a conversation with {poss_pronoun} {partner_role}. {name} says this about the {partner_role} {pronoun} is talking to: "{detail}".  This is the conversation they have so far:
    CONVERSATION:
    {conversation}

    Write six diverse dialogs that {name} can say next in the conversation with {poss_pronoun} {partner_role} based on the following keywords and context i.e., the things that {name} wants to talk about in the conversation. Please also take into account that {name} has an acquaintance of {level} on a scale of 10 with {partner_role} and is currently feeling {emotion}. Generate your answer in the form of a json list where each entry is a string. Start and end your answer with square brackets.

    KEYWORDS: {keywords}
    CONTEXT: {context}
    DIALOG: 
    """

    if len(conversation) == 0:
        return prompt_no_conv.format(name=name, age=age, gender=gender, pronoun=pronoun, poss_pronoun=poss_pronoun, partner_role=partner, detail=detail, level=level, emotion=emotion, keywords=keywords)
    else:
        if context is None or len(context) == 0:
            return prompt_with_conv.format(name=name, age=age, gender=gender, pronoun=pronoun, poss_pronoun=poss_pronoun, partner_role=partner, detail=detail, level=level, emotion=emotion, keywords=keywords, conversation=conversation_string)
        else:
            return prompt_with_conv_and_context.format(name=name, age=age, gender=gender, pronoun=pronoun, poss_pronoun=poss_pronoun, partner_role=partner, detail=detail, level=level, emotion=emotion, keywords=keywords, conversation=conversation_string, context=context)

def get_chatgpt_output(prompt):
    dialogues = ['']  # Initialize dialogues to a default value
    messages = [{"role": "system", 
                 "content": "You are a intelligent assistant."}]
                
    messages.append(
        {"role": "user", "content": prompt}
    )
    chat = client.chat.completions.create(
        model="gpt-3.5-turbo", 
        messages=messages
    )
    reply = chat.choices[0].message.content

    print(" ***** Decoding ****** ")
    print(reply)

    try:
        dialogues = json.loads(reply)
    except:
        print(f"Generation failed!")
        cleaned_reply = reply.replace("json", "").replace("```", "")
        dialogues = json.loads(cleaned_reply)

    if type(dialogues[0]) == dict:
        dialogues = list(dialogues[0].values())

    if len(dialogues) == 1:
        dialogues = dialogues[0].split(', ')

    return dialogues

def get_llava_keyword_output(image_path):

    image = Image.open(image_path)
    print("Running LLaVA model")
    conversation_string = ""
    for d in conversation:
        conversation_string += d["speaker"] + ' says ,' + d["text"] + "\n"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # prompt = "<image>\nUSER: Based on the given image, write SIX diverse keywords for dialogues that User would say next in the following conversation.\n\n%s\nASSISTANT: " % conversation_string
    prompt = "<image>\nUSER: The following conversation takes place between two people.\n\n%s\n User shares this image as response in the conversation. Describe the image in the context of the conversation. ASSISTANT: " % conversation_string
    print(prompt)
    inputs = llava_processor(text=prompt, images=image, return_tensors="pt").to(device)
    print("Generating")
    generate_ids = llava_model.generate(**inputs, max_new_tokens=100)
    output = llava_processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False, return_full_text=False)[0]
    print(output)
    output = output.split('ASSISTANT:')[-1].strip().replace('Speaker 1 says,', '').replace('Speaker 2 says,', '').replace('User says,', '').replace('Partner says,', '')
    print(output)
    curr_turn['image_caption'] = output
    keyword_prompt = ("Generate six keywords that can be used to generate the next dialogue by the user in a conversation based on a given image."
            "In the conversation, there are two speakers: the user and the partner."
            "Below are the provided basic information of a user's persona, and the relationship with the partner."
            "Please also take the acquaintance level into account for generating related keywords."
            "(If the level is higher, the keywords might related to more personal topics because they are closer.)"
            "For example:\n"
            "USER PERSONA: pronouns - she/her, age - 24"
            "PARTNER RELATIONSHIP: physician, Dr. Johnson has been my primary care physician for nearly 3 years, acquaintance level - 8/10"
            "IMAGE CAPTION: In the image, a pair of hands are shown with a bandage on the thumb of the left hand. The hands seem to swollen slightly."
            "KEYWORDS: hand swell, bandage change, hand bandage, pain hand, can't do work pain, hand muscle sore\n"
            "Generate the list of keywords in the form of a json list. Start and end your answer in a square bracket."
            "[NOTICE!] Generate keywords for the next dialogue in the conversation based on the image caption."
            "USER PERSONA: pronouns - {gender}, age - {age}"
            "PARTNER RELATIONSHIP: {partner}, {detail}, acquaintance level - {level}/10"
            "CONVERSATION: {conversation_string}"
            "IMAGE CAPTION: {caption}"
            "KEYWORDS: ")
    output_list = get_chatgpt_output(keyword_prompt.format(gender=pronouns, age=age, partner=partner, detail=detail, level=level, conversation_string=conversation_string, caption=output))
    print(f'image-based keywords: {output_list}')
    return output_list

old_element_list = ["", "", "", "", "", ""]
def save_topic(topic):
    try: 
        index = old_element_list.index("")
    except: 
        index = None

    if index is not None:
        old_element_list[index] = topic
    else:
        old_element_list.pop(0)
        old_element_list.append(topic)
    return old_element_list

def get_selected(input):
    return input

def respond(message, chat_history):
    chat_history = chat_history + [(message, None)]

    if curr_turn == {}:
        conversation.append({'speaker': 'User', 'text': message})
    else:
        curr_turn['speaker'] = 'User'
        curr_turn['text'] = message
        conversation.append(curr_turn.copy())
        curr_turn.clear()
    
    return message, chat_history

def add_file(file, chat_history):
    chat_history = chat_history + [((file.name,), None)]
    file_path = file.name 
    keywords = get_llava_keyword_output(file_path)
    return [chat_history] + keywords

transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")
def transcribe(audio):
    sr, y = audio
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    return transcriber({"sampling_rate": sr, "raw": y})["text"]

def submit_audio(audio_data, chat_history):
    print(audio_data)
    transcription = transcribe(audio_data).replace('little bit of a ', '')
    conversation.append({'speaker': 'Partner', 'text': transcription})
    curr_turn.clear()
    chat_history = chat_history + [(None, transcription)]
    clear_audio()
    return chat_history, None, ""

def clear_audio():
    return None, ""


with gr.Blocks(js=js) as app:
    with gr.Column() as user_persona:
        with gr.Row():
            with gr.Column():
                gr.Textbox(visible=False)
            with gr.Column():
                gr.Markdown("## User Persona")
                name = gr.Textbox(label="Name")
                pronouns = gr.CheckboxGroup(choices=["He/him", "She/her", "They/them"], label="Pronouns")
                age = gr.Dropdown([i for i in range(0, 151)], label="Age")
                persona_btn = gr.Button("Continue")
            with gr.Column():
                gr.Textbox(visible=False)

    with gr.Column(visible=False) as acquaintance_level:
        with gr.Row():
            with gr.Column():
                gr.Textbox(visible=False)
            with gr.Column():
                gr.Markdown("## Define Relationship with Communication Partner")
                partner = gr.Textbox(label="Role of Communication Partner", info="Enter the role of your communication partner e.g., physician, friend, parent, colleague etc.")
                detail = gr.Textbox(label="Detail", info="Enter specific details about communication partner: expertise (if physician), or relationship with aided speaker (if family/friend). E.g., 'Anne lives on the same street and we talk daily.'.")
                level = gr.Slider(1, 10, value=5, label="Acquaintance Level", info="How well you know the speaker? Choose from 1 (stranger) to 10 (well-known partner).")
                level_btn = gr.Button("Continue")
            with gr.Column():
                gr.Textbox(visible=False)

    with gr.Column(visible=False) as upload_file:
        with gr.Row():
            with gr.Column():
                gr.Button(visible=False)
            with gr.Column():
                gr.Markdown("## Enter Context for the Conversation")
                context = gr.Textbox(label="Context", info="Information entered in this box will be used to generate relevant dialogues.")
            with gr.Column():
                gr.Button(visible=False)
        with gr.Row():
            with gr.Column():
                gr.Button(visible=False)
            with gr.Column():
                upload_btn = gr.Button("Let's Chat!")
            with gr.Column():
                gr.Button(visible=False)
    
    old_topic_list = []
    topic_list = []
    dialogue_list = []
    freq_list = []
    with gr.Column(visible=False) as chat_page:
        with gr.Column():
            gr.Button(visible=False)

        with gr.Row():
            chatbot = gr.Chatbot()

            with gr.Column():
                with gr.Tab("Keywords"):
                    gr.Markdown("`OLD`")
                    with gr.Row():
                        for _ in range(2):
                            old_btn = gr.Button(visible=True, value="")
                            old_topic_list.append(old_btn)
                        for _ in range(2):
                            old_btn = gr.Button(visible=True, value="")
                            old_topic_list.append(old_btn)
                        for _ in range(2):
                            old_btn = gr.Button(visible=True, value="")
                            old_topic_list.append(old_btn)

                    gr.Markdown("`NEW`")
                    with gr.Row():
                        for _ in range(2):
                            topic_btn = gr.Button(visible=True, value="")
                            topic_list.append(topic_btn)
                        for _ in range(2):
                            topic_btn = gr.Button(visible=True, value="")
                            topic_list.append(topic_btn)
                        for _ in range(2):
                            topic_btn = gr.Button(visible=True, value="")
                            topic_list.append(topic_btn)

                        suggest_topic_btn = gr.Button(visible=True, value="Suggest Keywords")
                        suggest_topic_btn.click(update_topics, inputs=[pronouns, age, partner, detail, level, context], outputs=topic_list)

            with gr.Column():
                with gr.Row():
                    gr.Markdown("If you want to clear all inputs and start a new conversation, press the button:")
                with gr.Row():
                    reload_btn = gr.Button("Clear and Reload", scale=1)
                    # reload_btn.click(None, [], [], js="() => {window.location.reload();}")
                    reload_btn.click(None, [], [])
                with gr.Row():
                    gr.Markdown("Upload image or file to share or add context")
                with gr.Row():
                    file_btn = gr.UploadButton("📁", file_types=["image"])
                    file_btn.upload(add_file, inputs=[file_btn, chatbot], outputs=[chatbot] + topic_list)
                with gr.Row():
                    emotion = gr.Radio(["Excited", "Happy", "Anxious", "Scared", "Sad", "Angry"], value="Emotion", info="Use this emotion in my dialogue", interactive=True)
                with gr.Row():
                    gr.ClearButton([emotion], value="Clear Emotion")
        
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    audio_input = gr.Audio(sources=["microphone"])
                with gr.Row():
                    clear_btn = gr.Button("Clear")
                    audio_btn = gr.Button("Submit")
                    audio_output = gr.Textbox(visible=False)
                    clear_btn.click(clear_audio, inputs=[], outputs=[audio_input, audio_output])

            with gr.Column():
                with gr.Row():
                    topic_input = gr.Textbox(label="Keywords", value="type keyword...", interactive=True, scale=1)
                    topic_typein_btn = gr.Button("Suggest Dialogues", scale=0)

            with gr.Column():
                with gr.Row():
                    dialogue_input = gr.Textbox(label="Dialogue", value="type dialogue...", interactive=True, scale=1)
                    dialogue_typein_btn = gr.Button("Enter Dialogue", scale=0, elem_id="dialogue_typein_btn")
                    dialogue_input.submit(respond, [dialogue_input, chatbot], [dialogue_input, chatbot])
                    dialogue_typein_btn.click(respond, [dialogue_input, chatbot], [dialogue_input, chatbot])

        with gr.Row():
            with gr.Column():
                with gr.Tab("Frequently Used Words"):
                    with gr.Row():
                        freq_btn0 = gr.Button("OK.", elem_id=f"freq_0")
                        freq_btn1 = gr.Button("Hi!", elem_id=f"freq_1")
                        freq_list.append(freq_btn0)
                        freq_list.append(freq_btn1)
                    with gr.Row():
                        freq_btn2 = gr.Button("Yes.", elem_id=f"freq_2")
                        freq_btn3 = gr.Button("No.", elem_id=f"freq_3")
                        freq_list.append(freq_btn2)
                        freq_list.append(freq_btn3)
                    with gr.Row():
                        freq_btn4 = gr.Button("Please.", elem_id=f"freq_4")
                        freq_btn5 = gr.Button("Thanks.", elem_id=f"freq_5")
                        freq_list.append(freq_btn4)
                        freq_list.append(freq_btn5)

                    for i in range(len(freq_list)):
                        freq_list[i].click(respond, inputs=[freq_list[i], chatbot], outputs=[freq_list[i], chatbot])

            with gr.Column(scale=2):
                with gr.Tab("Dialogues"):
                    with gr.Row():
                        for i in [0, 1]:
                            dialogue_btn = gr.Button(visible=True, value="", elem_id=f"res_{i}")
                            dialogue_list.append(dialogue_btn)
                    with gr.Row():
                        for i in [2, 3]:
                            dialogue_btn = gr.Button(visible=True, value="", elem_id=f"res_{i}")
                            dialogue_list.append(dialogue_btn)
                    with gr.Row():
                        for i in [4, 5]:
                            dialogue_btn = gr.Button(visible=True, value="", elem_id=f"res_{i}")
                            dialogue_list.append(dialogue_btn)
                    with gr.Row():
                        with gr.Column():
                            more_btn = gr.Button("Suggest more dialogues!")
                            selected_btn = gr.Button(visible=False, value="hello world")
                    
                    # Show topic in type in textbox
                    for i in range(len(topic_list)):
                        topic_list[i].click(get_selected, inputs=topic_list[i], outputs=topic_input)

                    # Generate dialogues from type in topic
                    topic_input.submit(generate_dialogues, inputs=[topic_input, context, pronouns, age, partner, detail, level], outputs=dialogue_list)
                    topic_input.submit(save_topic, inputs=topic_input, outputs=old_topic_list)
                    topic_input.submit(get_selected, inputs=topic_input, outputs=selected_btn)
                    topic_typein_btn.click(generate_dialogues, inputs=[topic_input, context, pronouns, age, partner, detail, level], outputs=dialogue_list)
                    topic_typein_btn.click(save_topic, inputs=topic_input, outputs=old_topic_list)
                    topic_typein_btn.click(get_selected, inputs=topic_input, outputs=selected_btn)

                    # Generate more dialogues if needed
                    more_btn.click(generate_dialogues, inputs=[selected_btn, context, pronouns, age, partner, detail, level], outputs=dialogue_list)

                    # Show dialogue in type in textbox
                    for i in range(len(dialogue_list)):
                        dialogue_list[i].click(get_selected, inputs=dialogue_list[i], outputs=dialogue_input)
                    
                    # Show old topic in type in textbox
                    for i in range(len(old_topic_list)):
                        old_topic_list[i].click(get_selected, inputs=old_topic_list[i], outputs=topic_input)

    audio_btn.click(submit_audio, inputs=[audio_input, chatbot], outputs=[chatbot, audio_input, audio_output]).then(update_topics, inputs=[pronouns, age, partner, detail, level, context], outputs=topic_list).then(generate_dialogues, inputs=None, outputs=dialogue_list)
    persona_btn.click(fn=switch2role, inputs=[name, pronouns, age], outputs=[user_persona, acquaintance_level])
    level_btn.click(fn=switch2upload, inputs=[level], outputs=[acquaintance_level, upload_file])
    upload_btn.click(fn=switch2chat, inputs=[context], outputs=[upload_file, chat_page])
    upload_btn.click(fn=update_topics, inputs=[pronouns, age, partner, detail, level, context], outputs=topic_list)


app.launch(share=True, server_port=7862)
