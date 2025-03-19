# LLM to Enhance Communication for AAC Speakers

This project provides a system designed to help users with communication needs. The system generates contextually appropriate dialogue suggestions based on user persona, relationship with the communication partner, and conversation context.

## Overview

The system uses a combination of large language models (LLMs) and image processing capabilities to:

1. Generate context-aware keywords for potential dialogue options
2. Create natural-sounding dialogue suggestions based on selected keywords
3. Provide frequently used phrases for quick access
4. Process and incorporate shared images into the conversation context
5. Support voice input through speech recognition

## Features

- **Persona-based dialogue**: Customizable user persona with name, age, and pronouns
- **Relationship modeling**: Define the role and acquaintance level with the communication partner
- **Context-aware suggestions**: Keywords and dialogues adapt to the ongoing conversation
- **Multi-modal input**: Support for text, image, and voice input
- **Emotion selection**: Express different emotions in generated dialogues
- **Frequently used words**: Quick access to common phrases

## Components

The system consists of three main Python files:

1. `app.py`: The main application with Gradio UI implementation
2. `conversational_agent.py`: Functions for dialogue generation and image processing
3. `global_methods.py`: Utility functions for working with language models
4. `templates.py`: Prompt templates for language model interactions

## Installation

1. Clone the repository:
```bash
git clone https://github.com/lilyyang1998/LLM-4-AAC.git
cd LLM-4-AAC
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key"
```

## Usage

1. Start the application:
```bash
python app.py
```

2. Access the web interface at `http://localhost:7862`

3. Set up your persona:
   - Enter your name, pronouns, and age
   - Define the role of your communication partner and your acquaintance level
   - Optionally add context information for the conversation

4. Use the system:
   - Select keywords to generate relevant dialogue options
   - Use the provided dialogue buttons or type your own message
   - Upload images to incorporate visual context
   - Record voice input from your communication partner
   - Express different emotions using the emotion radio buttons

## Conversation Flow

1. Define your persona (name, pronouns, age)
2. Specify your relationship with the communication partner (role, details, acquaintance level)
3. Optionally provide context for the conversation
4. Begin the conversation with suggested dialogue options
5. Continue the conversation with AI-generated suggestions based on the evolving context

## Demo Youtube Link
[https://youtu.be/BqKLk9aNUrE?feature=shared](https://youtu.be/BqKLk9aNUrE?feature=shared)