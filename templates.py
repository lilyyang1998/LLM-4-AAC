PERSONA_FROM_CHOICES = "You are {name}, a {age}-years old {gender} with the following persona: {persona}"

CONVERSATION2FACTS_PROMPT = """
Write a concise and short list of all possible OBSERVATIONS about {user_name} and {speaker_name} that can be gathered from the CONVERSATION. Each OBSERVATION should contain a piece of information about the speaker. The OBSERVATIONS should be objective factual information about the speaker that can be used as a database about them. Avoid abstract observations about the dynamics between the two speakers such as 'speaker is supportive', 'speaker appreciates' etc. Do not leave out any information from the CONVERSATION. Important: Escape all double-quote characters within string output with backslash.\n\n
"""

REFLECTION_INIT_PROMPT = "{}\n\nGiven the information above, what are the three most salient insights that {} has about {}? Give concise answers in the form of a json list where each entry is a string."

REFLECTION_CONTINUE_PROMPT = "{} has the following insights about {} from previous interactions.{}\n\nTheir next conversation is as follows:\n\n{}\n\nGiven the information above, what are the three most salient insights that {} has about {} now? Give concise answers in the form of a json list where each entry is a string."

SELF_REFLECTION_INIT_PROMPT = "{}\n\nGiven the information above, what are the three most salient insights that {} has about self? Give concise answers in the form of a json list where each entry is a string."

SELF_REFLECTION_CONTINUE_PROMPT = "{} has the following insights about self.{}\n\n{}\n\nGiven the information above, what are the three most salient insights that {} has about self now? Give concise answers in the form of a json list where each entry is a string."

AGENT_CONV_PROMPT_SESS_1 = "The current time and date are {time_date}. You are {persona}. You are initiating a conversation with {speaker}. Write the next thing you would say to {speaker} in the conversation. Write replies in less than 20 words\n\nCONVERSATION:\n\n"

AGENT_CONV_PROMPT_SESS_1_1 = "You are {persona}. The current time and date are {time_date}. You are in a conversation with {speaker}. Write the next thing you would say to {speaker} in the following conversation. Write replies in less than 20 words\n\nCONVERSATION:\n\n"

AGENT_CONV_PROMPT_SESS_1_w_CONTEXT = "The current time and date are {time_date}. You are {persona}. You are initiating a conversation with {speaker}. Write the next thing you would say to {speaker} in the conversation. Write replies in less than 20 words\n\nCONVERSATION:\n\nUse the following INFO to write the next RESPONSE in the conversation.\nINFO: {info}\n\nRESPONSE:"

AGENT_CONV_PROMPT_SESS_1_1_w_CONTEXT = "You are {persona}. The current time and date are {time_date}. You are in a conversation with {speaker}. Write the next thing you would say to {speaker} in the following conversation. Write replies in less than 20 words\n\nCONVERSATION:\n\nUse the following INFO to write the next RESPONSE in the conversation.\nINFO: {info}\n\nRESPONSE:"

AGENT_CONV_PROMPT_gt1_1 = "You are {persona}. {summary_history} The current time and date are {time_date}. Now, you are initiating a conversation with {speaker}. Write the next thing you would say to {speaker} in the conversation. Write replies in less than 20 words\n\nCONVERSATION:\n\n"

AGENT_CONV_PROMPT_gt1 = "You are {persona}. {summary_history} The current time and date are {time_date}. You are initiating a conversation with {speaker}. Write the next thing you would say to {speaker} in the conversation. Write replies in less than 20 words\n\nCONVERSATION:\n\n"

AGENT_CONV_PROMPT_gt1_1_w_CONTEXT = "You are {persona}. {summary_history} The current time and date are {time_date}. Now, you are initiating a conversation with {speaker}. Write the next thing you would say to {speaker} in the conversation. Write replies in less than 20 words\n\nCONVERSATION:\n\n"

AGENT_CONV_PROMPT_gt1_w_CONTEXT = "You are {persona}. {summary_history} The current time and date are {time_date}. You are initiating a conversation with {speaker}. Write the next thing you would say to {speaker} in the conversation. Write replies in less than 20 words\n\nCONVERSATION:\n\n"

# If starting the conversation, start with asking about their day or talking about something that happened in your life recently.