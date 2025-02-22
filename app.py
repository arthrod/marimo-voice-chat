# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "google-generativeai==0.8.4",
#     "gtts==2.5.4",
#     "marimo",
#     "modal==0.73.64",
#     "openai==1.63.2",
#     "python-dotenv==1.0.1",
#     "websocket-client==1.8.0",
# ]
# ///

import marimo

__generated_with = "0.11.6"
app = marimo.App(width="medium")


@app.cell
def _():
    (
        """
    CELL WITH INSTRUCTIONS
    To ensure I provide the most effective implementation details, could you clarify:

    What framework or environment are you using for the Marimo-based chatbot? Are you looking for a Python-based Marimo implementation or something else?
    Do you have any preferred text-to-speech (TTS) and speech-to-text (STT) libraries? For example, Google Speech API, Whisper, or pyttsx3 for voice synthesis?
    For the multilingual feature, should it be a simple button/toggle that switches between two predefined languages (e.g., English and Spanish), or should it support multiple language options?
    Should the chatbot support predefined responses, or do you want it to be AI-driven with an LLM backend (e.g., OpenAI's API)?
    Do you need the chatbot interface to be visually customized in any way, or is a simple functional implementation sufficient?
    Once I have these details, I’ll conduct the research and provide you with a full working implementation.






    You said:
    1. Marimo python. 2. Openai api. 3. Simple toggle. 4.openai api. 5. Customizable but at least the first version must work out of the box.
    ChatGPT said:
    Got it! I will research how to implement a simple chatbot using Marimo (Python) with OpenAI's API, including microphone input for speech recognition, text-to-speech functionality, and a simple toggle for switching between English and another language. The implementation will be functional out of the box but customizable for further improvements.

    I'll provide you with a full working code once the research is complete.

    Research completed in 5m · 19 sources
    Introduction
    Building a voice-interactive chatbot in Python is now easier with Marimo, an open-source reactive notebook platform. Marimo provides a ready-made chat UI component that can embed a custom chatbot interface directly into your notebook or app with just a few lines of code​
    SIMONWILLISON.NET
    ​
    SIMONWILLISON.NET
    . We’ll leverage this along with OpenAI’s API to create a simple chatbot that can handle both text and speech. The chatbot will support:
    Text and Voice Input: Users can type messages or press a microphone button and speak.
    AI Responses via OpenAI: The user’s input is sent to an OpenAI GPT model (e.g. GPT-3.5) to generate a reply.
    Text and Voice Output: The reply is displayed in the chat interface and simultaneously synthesized to speech so the user can listen.
    Language Toggle: A toggle switch lets users switch between English and another language (predefined) for both input and output, without complex internationalization.
    This guide will walk through the implementation step by step – from setting up the environment to integrating speech-to-text and text-to-speech – and culminate in a complete code example. Each section includes code snippets and explanations for seamless integration, ensuring the final solution works out of the box and is easy to customize.
    Prerequisites and Setup
    Before coding the chatbot, make sure you have the following:
    Python Environment: Ensure you have Python 3 installed. Marimo works within a Python environment (and even has an online playground​
    MARIMO.IO
    ), but you can also use it locally by installing the package.
    Marimo: Install Marimo, which provides the reactive notebook and UI components. You can install it via pip:
    bash
    Copy
    pip install marimo
    Marimo is a reactive Python notebook environment​
    MARIMO.IO
    , so you can run this code in a Marimo notebook or as a Marimo app.
    OpenAI API Key and SDK: Sign up for an OpenAI API key (from the OpenAI dashboard) and install the OpenAI Python library:
    bash
    Copy
    pip install openai
    This will let us call OpenAI’s GPT (for chat completion) and Whisper (for speech-to-text) endpoints.
    gTTS (Google Text-to-Speech): Install the gTTS library for text-to-speech conversion:
    bash
    Copy
    pip install gTTS
    gTTS uses Google’s Text-to-Speech to synthesize speech from text, which we’ll use to speak the chatbot’s responses.
    Note: In a Marimo notebook, you may also declare these as requirements. For simplicity, we assume you can use pip to get the needed packages. Once the environment is set, import the necessary libraries in your Python code (Marimo, openai, gTTS, etc.) and set your OpenAI API key (e.g. openai.api_key = "YOUR_API_KEY").
    Creating the Chat Interface with Marimo
    Marimo’s UI module (marimo.ui) provides a high-level chat element that handles the chat history and user input box for you​
    SIMONWILLISON.NET
    . You just need to provide a function (or model) that generates a response given the conversation so far. Marimo will:
    Display the conversation in a chat bubble format (user and assistant messages).
    Provide a text input box for the user to type new messages.
    Call your supplied function whenever a new user message is sent, passing in the message history, and then display the function’s return value as the assistant’s reply​
    SIMONWILLISON.NET
    .
    Using mo.ui.chat: We initialize the chat interface by calling mo.ui.chat(response_fn), where response_fn is our Python function to handle incoming messages and produce a reply. For example, a trivial echo chatbot in Marimo could be:
    python
    Copy
    import marimo as mo

    def echo_model(messages, config=None):
        # Simply echo back the last user message
        return f"Echo: {messages[-1].content}"

    chat_ui = mo.ui.chat(echo_model)
    chat_ui
    In this echo_model, messages is a list of ChatMessage objects (each has a .role like "user" or "assistant" and .content text​
    DOCS.MARIMO.IO
    ). We return a string, and Marimo will add that as an assistant message in the chat. This minimal example shows how easy it is to embed a chat in the UI​
    SIMONWILLISON.NET
    .For our chatbot, we want response_fn to call OpenAI’s API. Marimo actually provides built-in chat handlers for popular LLM providers (OpenAI, Anthropic, etc.) which you can use directly​
    SIMONWILLISON.NET
    . For instance, you could do:
    python
    Copy
    chat_ui = mo.ui.chat(
        mo.ai.llm.openai(model="gpt-3.5-turbo", api_key="YOUR_API_KEY", system_message="You are a helpful assistant.")
    )
    This uses Marimo’s built-in OpenAI connector (with a specified model and system prompt) to handle the chat responses​
    SIMONWILLISON.NET
    . However, to integrate custom features like our language toggle and audio output, we’ll write our own function using the OpenAI library directly. This gives us more control.Implementing the Response Function with OpenAI:
    Our response_fn will use OpenAI’s Chat Completion API (GPT-3.5 or GPT-4) to generate answers. Key steps in the function:
    Collect the conversation: We’ll take the messages list provided by Marimo (which contains the conversation history) and convert it into the format OpenAI’s API expects. This means a list of dicts like {"role": "user", "content": "..."} or {"role": "assistant", "content": "..."}. We can also inject a system message at the start to guide the assistant’s behavior (e.g. define its persona or language).
    Include language preference: If the user toggled the language switch (explained later), and for example chose Spanish, we can set the system message to “You are a helpful assistant. Respond in Spanish.” (Otherwise, it can be “Respond in English” or simply “You are a helpful assistant.”). This ensures the AI replies in the chosen language.
    Call OpenAI API: Use openai.ChatCompletion.create() with the model (e.g. "gpt-3.5-turbo"), your assembled message list, and get the assistant’s reply text.
    Return the reply: We’ll return the reply text so that Marimo displays it in the chat. But we actually want to return both text and an audio player for the reply, so in the next section we’ll modify this to include TTS output.
    Here’s a simplified version of the response function focusing on text:
    python
    Copy
    import openai

    openai.api_key = "YOUR_API_KEY"  # set your API key

    def chatbot_response(messages):
        # Determine system prompt based on language toggle (English/Spanish)
        if language_switch.value:  # Suppose True means Spanish mode
            system_msg = {"role": "system", "content": "You are a helpful assistant. Please respond in Spanish."}
        else:
            system_msg = {"role": "system", "content": "You are a helpful assistant. Respond in English."}
        # Build the message list for OpenAI (include system + conversation so far)
        conversation = [system_msg]
        for msg in messages:
            conversation.append({"role": msg.role, "content": msg.content})
        # Call OpenAI ChatCompletion
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=conversation)
        answer = response.choices[0].message.content
        return answer  # This will be displayed as the assistant's reply
    }

    chat_ui = mo.ui.chat(chatbot_response)
    In this snippet, language_switch is a UI element (to be defined later) that holds a boolean indicating the language choice. We prepend a system message accordingly. The OpenAI API call uses the entire conversation (including the latest user query) to generate a context-aware response. The returned answer string gets rendered as the assistant’s message in the chat UI.Note: The OpenAI ChatCompletion API expects conversation history for context. By including the prior messages (which Marimo maintains), the assistant can have a memory of the chat. The system message sets the default language and behavior. Also, ensure you handle the API key securely (you might use openai.api_key or an environment variable).
    Integrating Microphone Input (Speech-to-Text)
    To let users speak to the chatbot, we use Marimo’s Microphone UI element. The mo.ui.microphone() component renders a microphone button in the interface. When clicked, it will record audio from the user’s browser (after prompting for permission)​
    DOCS.MARIMO.IO
    . Once the recording is done, the audio data is available in Python as a binary blob (an io.BytesIO object)​
    DOCS.MARIMO.IO
    .Using mo.ui.microphone:
    We create a microphone element and specify an on_change callback. This callback function will be automatically invoked when a new audio recording is finished (i.e., when the microphone input value changes). The callback receives the audio data (as a BytesIO stream) as its argument. Our plan for the callback:
    Transcribe audio to text: Send the recorded audio to OpenAI’s Whisper API for transcription (speech-to-text). The OpenAI Python SDK provides openai.Audio.transcribe() to handle this using the Whisper model. We need to provide the model name (e.g., "whisper-1") and the audio file or file-like object. We’ll use the BytesIO from the mic directly.
    Handle audio format: Ensure the audio data is in a supported format and the API knows the format. Whisper API supports common formats like mp3, wav, m4a, webm, etc​
    DATACAMP.COM
    ​
    DATACAMP.COM
    . Marimo’s mic likely produces a WebM or WAV. To be safe, we can set the name attribute of the BytesIO object (e.g., audio_bytes.name = "voice.webm") so that the OpenAI SDK can infer the format​
    COMMUNITY.OPENAI.COM
    .
    Get the text result: The openai.Audio.transcribe() call returns the transcribed text (or a dictionary with the text). We extract the text string. For better accuracy, we can also specify the language to Whisper if we know it (OpenAI allows a language parameter​
    DATACAMP.COM
    ). For example, if the user selected Spanish mode, we might call openai.Audio.transcribe("whisper-1", audio_bytes, language="es"). Whisper can auto-detect language, but providing a hint can improve speed/accuracy​
    DATACAMP.COM
    .
    Submit as chat message: Take the transcribed text and inject it into the chat interface as if the user had typed it. Marimo’s chat element has a method send_message() that allows the backend to programmatically send a message into the chat UI​
    DOCS.MARIMO.IO
    . We’ll use chat_ui.send_message({"role": "user", "content": user_text}) to add the transcribed text as a user message. This will trigger our chatbot_response function to generate a reply (just like a normal user text input).
    Now, let’s implement the microphone and its callback:
    python
    Copy
    # Create a microphone UI element with a callback
    def transcribe_and_send(audio_bytes: io.BytesIO):
        # Ensure the audio BytesIO has a filename (extension) for format detection
        audio_bytes.name = "voice.webm"  # assuming webm/opus from browser; use "wav" if needed
        # Transcribe speech to text using OpenAI Whisper
        transcript = openai.Audio.transcribe(model="whisper-1", file=audio_bytes)
        # Extract the text (the API may return an object or dict with 'text')
        user_text = transcript.get('text') if isinstance(transcript, dict) else str(transcript)
        user_text = user_text.strip()
        if user_text:
            # Send this as a new user message in the chat UI
            chat_ui.send_message({"role": "user", "content": user_text})

    mic_button = mo.ui.microphone(label="🎤 Speak", on_change=transcribe_and_send)
    In this code:
    We label the microphone button with a microphone emoji and text for clarity.
    The transcribe_and_send function does the work of calling Whisper. We set audio_bytes.name so that the OpenAI SDK treats it as a file named “voice.webm” (WebM is a supported format​
    DATACAMP.COM
    ). Alternatively, if the mic yields WAV, use .name = "audio.wav". This little trick is needed because the OpenAI SDK infers the file type from the file name extension​
    COMMUNITY.OPENAI.COM
    .
    We call openai.Audio.transcribe("whisper-1", file=audio_bytes) and get the result. (If using an older version of openai SDK, the function might be openai.Audio.transcribe("whisper-1", audio_bytes) without explicit file= parameter.) We can also pass language="en" or "es" based on the toggle to guide the transcription​
    DATACAMP.COM
    .
    Once we have user_text, we use chat_ui.send_message() to inject it. The dict requires a "role" ("user" in this case) and the "content" string. Marimo will add this to the chat and call our chatbot_response function, resulting in the AI’s answer appearing.
    Now the user can click the mic button, speak, and the spoken words will be transcribed and appear as a chat message from the user, prompting the AI to respond. This happens almost as seamlessly as typing – the user doesn’t manually type the transcribed text; it’s done behind the scenes.
    Adding Text-to-Speech for Responses
    We want the chatbot’s replies not only in text, but also spoken aloud (or at least available as audio). To do this, we integrate a text-to-speech (TTS) step in the response generation. We’ll use gTTS (Google Text-to-Speech), a simple Python library that converts text to spoken audio.Using gTTS for TTS:
    gTTS can take a text string and a language code, and produce an audio file (MP3 by default) with the spoken version. Basic usage example from documentation:
    python
    Copy
    from gtts import gTTS
    text = "Hello, this is a text-to-speech example."
    tts = gTTS(text=text, lang='en', slow=False)
    tts.save("output.mp3")  # saves the speech audio to a file
    STACKOVERFLOW.COM
    . In our case, we don’t want to save to a file on disk; instead, we’ll use an in-memory file. gTTS provides a write_to_fp() method that writes the MP3 bytes to a file-like object​
    GTTS.READTHEDOCS.IO
    . We can create an io.BytesIO() buffer and write the audio data into it. This buffer can then be used to play the audio in the Marimo interface.Embedding Audio in Marimo:
    Marimo has an mo.audio() element that can take a source (URL or BytesIO) and render an HTML audio player​
    DOCS.MARIMO.IO
    . We will use mo.audio(bytes_buffer) to create an audio player for the chatbot’s reply. This player will appear in the chat bubble along with the text, so the user can immediately play the reply voice. (Depending on browser settings, the user might need to click play since auto-play of audio might be restricted. The audio element will at least provide the output without additional setup.)Integrating TTS in the Response Function:
    We modify our chatbot_response function to do the following when producing the assistant’s answer:
    After getting the answer text from OpenAI, generate speech audio from it. We choose the language for TTS based on the toggle (e.g., use 'en' for English or 'es' for Spanish so the accent/pronunciation is correct).
    Create a gTTS object with the answer text and language, and write it to a BytesIO buffer.
    Reset the buffer’s pointer to start (buffer.seek(0)) so it can be read from the beginning.
    Use mo.vstack([...]) to return a combined UI element containing both the answer text and an audio player. We stack a text element (we can use mo.md(answer_text) for nice formatting) above the audio player. Using a vertical stack ensures they appear together as one message​
    DOCS.MARIMO.IO
    .
    Here’s how the updated response function looks with TTS:
    python
    Copy
    import io
    from gtts import gTTS
    import marimo as mo

    def chatbot_response(messages):
        # Determine target language code from the toggle (False = English, True = Spanish)
        target_lang = "es" if language_switch.value else "en"
        system_msg = {"role": "system", "content": "You are a helpful assistant. Respond in Spanish." if target_lang=="es" 
                                                      else "You are a helpful assistant. Respond in English."}
        # Assemble conversation for OpenAI
        convo = [system_msg] + [{"role": m.role, "content": m.content} for m in messages]
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=convo)
        answer_text = response.choices[0].message.content.strip()
        # Text-to-speech conversion of the answer
        tts = gTTS(text=answer_text, lang=target_lang, slow=False)
        audio_data = io.BytesIO()
        tts.write_to_fp(audio_data)         # write MP3 data to buffer
        audio_data.seek(0)                  # reset buffer pointer
        # Return a combined element: Answer text + audio player
        return mo.vstack([ mo.md(answer_text), mo.audio(audio_data) ])
    Let’s break down the TTS part:
    We choose target_lang as "es" (Spanish) if the switch is on, otherwise "en" for English. gTTS supports many language codes (e.g., 'en', 'es', 'fr', etc.). The voice used will correspond to the language and a default accent (can be adjusted with a tld parameter if needed, e.g., tld="co.uk" for British English​
    GTTS.READTHEDOCS.IO
    , but we’ll keep it simple).
    gTTS(text=answer_text, lang=target_lang, slow=False) creates the TTS object. We use slow=False for normal speed.
    Instead of saving to a file, we do tts.write_to_fp(audio_data), where audio_data is a BytesIO buffer. This writes the MP3 bytes into audio_data​
    GTTS.READTHEDOCS.IO
    .
    We then ensure the buffer is at the beginning (audio_data.seek(0)).
    mo.audio(audio_data) creates an audio player element from this buffer. Marimo will handle converting it to an appropriate HTML audio tag in the output​
    DOCS.MARIMO.IO
    .
    We put the answer text (formatted via mo.md) and the audio element into a vertical stack container: mo.vstack([...]). According to Marimo’s docs, vstack will stack the given items vertically (one above the other)​
    DOCS.MARIMO.IO
    . This means in the chat bubble, the assistant’s response will show the text, and just below it, an audio bar to play the spoken answer. It will appear as a single assistant message containing both elements.
    Now our chat function returns a UI element (the stack), not just a string. Marimo’s chat is designed for this – it can handle messages that are text, images, plots, or other UI components as the response​
    SIMONWILLISON.NET
    . The user will see the text immediately and can click play to hear it.
    Implementing the Language Toggle
    The language toggle is a simple UI control to switch the interface between English and the second language (for example, Spanish). We don’t need a full i18n framework since we’re only dealing with two languages in a limited way. We will use a boolean switch that the user can flip.Using mo.ui.switch: Marimo offers a switch() component, which is essentially a toggle that holds a boolean value (False/True)​
    DOCS.MARIMO.IO
    . We can label this switch to indicate its function, e.g., “Spanish Mode” or “🇺🇸/🇪🇸” to show language. The switch value will be False for English (default) and True for Spanish in our design. We’ll refer to this in our code to decide on translation/transcription language and the assistant’s response language.We create the switch at the top-level so it’s always visible for the user to toggle. For example:
    python
    Copy
    language_switch = mo.ui.switch(label="Spanish Mode")
    By default this will render a toggle (perhaps a checkbox-style slider) labeled “Spanish Mode”. When unchecked (False), we treat it as English; when checked (True), we use Spanish. We already used language_switch.value inside our functions above to branch logic. Marimo’s reactive nature means whenever language_switch.value changes, it can trigger re-computation of dependent cells. In our case, the chat function will read its value each time it runs, so the next user query after toggling will use the updated language setting.What the toggle affects:
    OpenAI prompt: As shown, we change the system message to instruct the assistant to reply in the chosen language.
    Whisper transcription: We could pass the corresponding language code to Whisper for better accuracy (optional, since Whisper auto-detects).
    TTS voice: We pass the language code to gTTS so the spoken output uses the correct language/voice.
    Because we handle all these in code using the language_switch.value, the UI control itself is simple. We don’t need separate text labels for “English” vs “Spanish” because the system prompt and TTS handle it dynamically. The user just flips the switch to change the mode, and the next interactions will be in that language.
    Complete Code Example
    Below is the full code combining all the components – chat UI, OpenAI integration, microphone input, text-to-speech output, and language toggle. This code can be run in a Marimo notebook or app. When executed, it will display the chatbot interface with a toggle and a microphone button. Users can type or talk to the bot and get responses in text and audio.
    python
    Copy
    import marimo as mo  
    import openai, io  
    from gtts import gTTS  

    # Set your OpenAI API key  
    openai.api_key = "YOUR_OPENAI_API_KEY"  

    # Language toggle switch (False = English, True = Spanish)  
    language_switch = mo.ui.switch(label="Spanish Mode")  

    # Define the chatbot response function for mo.ui.chat  
    def chatbot_response(messages):  
        # Choose language based on toggle  
        use_spanish = language_switch.value  
        lang_code = "es" if use_spanish else "en"  
        # System prompt to enforce language  
        system_content = ( "You are a helpful assistant. Respond in Spanish."  
                           if use_spanish else  
                           "You are a helpful assistant. Respond in English." )  
        convo = [{"role": "system", "content": system_content}]  
        # Append conversation history (user and assistant messages so far)  
        for msg in messages:  
            convo.append({"role": msg.role, "content": msg.content})  
        # OpenAI ChatCompletion API call  
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=convo)  
        answer_text = response.choices[0].message.content.strip()  

        # Convert answer_text to speech audio  
        tts = gTTS(text=answer_text, lang=lang_code, slow=False)  
        audio_bytes = io.BytesIO()  
        tts.write_to_fp(audio_bytes)  
        audio_bytes.seek(0)  # reset pointer to beginning  

        # Return combined text + audio as the chat reply  
        return mo.vstack([  
            mo.md(answer_text),       # display answer text  
            mo.audio(audio_bytes)     # audio player for answer  
        ])  

    # Create the chat UI element with our response function  
    chat_ui = mo.ui.chat(chatbot_response)  

    # Microphone input with on_change callback for STT  
    def transcribe_and_send(audio_bytes):  
        # Name the BytesIO for format detection (assume webm audio)  
        audio_bytes.name = "speech.webm"  
        # Transcribe speech to text using OpenAI Whisper  
        transcript = openai.Audio.transcribe("whisper-1", file=audio_bytes)  
        text = transcript.get("text") if isinstance(transcript, dict) else str(transcript)  
        text = text.strip()  
        if text:  
            # Send the transcribed text as a user message to the chat  
            chat_ui.send_message({"role": "user", "content": text})  

    mic_button = mo.ui.microphone(label="🎤 Speak", on_change=transcribe_and_send)  

    # Layout: display the toggle, mic button, and chat UI in a vertical stack  
    mo.vstack([  
        mo.md("**Language:**"), language_switch,  
        mo.md("**Speak or Type your message**"), mic_button,  
        chat_ui  
    ])
    How it works: When you run this code, you’ll see a toggle switch for language, a “Speak” microphone button, and the chat interface. You can toggle the switch to your desired language at any time. To interact: either type a message in the chat input box and hit Enter, or click the microphone button and speak your question. If using the mic: after you stop recording, the audio will be sent to Whisper and the recognized text will appear as if you typed it. The OpenAI GPT model then generates a reply. The reply will appear in the chat with an audio player below it. You can read the text and click play to hear the spoken response.This fulfills the requirements: the user can converse by voice or text, and the assistant responds in both mediums. The code is modular and uses simple constructs, making it easy to extend. For example, you could swap out gTTS for another TTS service, or add more languages by replacing the switch with a dropdown (mo.ui.dropdown) of language options and adjusting the logic.
    Running and Customizing the Chatbot
    To run the chatbot, ensure you are in a Marimo environment (or have the Marimo app serving the UI). The interface elements (toggle, mic, chat) should render automatically from the code above. Here’s how a typical interaction would go:
    Select Language: Flip the Language switch to your desired language. (Off for English, on for Spanish in our example).
    Speak or Type: Click the 🎤 Speak button and record your question, or type a message into the chat input and send it. For instance, ask “What’s the weather today?” (in English mode) or “¿Cómo estará el clima hoy?” (in Spanish mode).
    Wait a moment: The audio (if used) is sent to OpenAI Whisper. Within a second or two, the transcribed text will appear in the chat as your message. The OpenAI model is then queried, which usually responds in a couple of seconds.
    Read and Listen to Reply: The chatbot’s answer appears as a chat bubble containing text and an audio player. For example, it might say in Spanish mode: “Según el pronóstico, hoy estará soleado con temperaturas agradables.” and an audio bar. Click the play button to hear it spoken aloud.
    Because we used streaming UI components, each step happens interactively. The design ensures simultaneous text and speech output – you don’t have to wait for the audio separately; it’s generated and available as soon as the text is displayed.Customization Options:
    Model and API: You can switch to a different OpenAI model (like gpt-4 if you have access) by changing the model name in ChatCompletion.create(). You could also integrate Anthropic’s Claude or other providers using Marimo’s built-in handlers​
    SIMONWILLISON.NET
    , but that would require their API keys and possibly different function calls.
    Language: Our toggle currently switches between English and Spanish. To change the second language, adjust the labels and language codes accordingly (e.g., to French 'fr' or German 'de'). For multiple languages, consider using a dropdown (mo.ui.dropdown or mo.ui.radio with options) instead of a simple switch.
    Speech Recognition: We used OpenAI’s Whisper API for transcription for simplicity and accuracy. If you want an offline solution, you might integrate a library like SpeechRecognition with a local engine, but that would require additional setup. Whisper can handle many languages and accents as shown in OpenAI’s documentation​
    DATACAMP.COM
    ​
    DATACAMP.COM
    .
    Text-to-Speech: gTTS was chosen for ease of use (no API key required and supports many languages). The trade-off is that it requires an internet connection to Google’s TTS service and returns an MP3. If you need offline TTS, you could use libraries like pyttsx3 (for a robotic voice) or use an API like Amazon Polly or Azure TTS for more natural voices. In Marimo, as long as you get an audio file or bytes, you can feed it to mo.audio().
    Finally, because Marimo is a reactive notebook, any changes to the code or UI state update the relevant parts automatically. This makes our chatbot quite responsive. We’ve avoided any complex internationalization frameworks – the code simply switches languages via conditionals. This straightforward approach is effective for a two-language scenario.References: The implementation draws on Marimo’s documentation and OpenAI’s API usage. Marimo’s chat UI and interactive elements (chat, microphone, switch, audio) enable the interface with minimal code​
    SIMONWILLISON.NET
    ​
    DOCS.MARIMO.IO
    . We used OpenAI’s Whisper for transcription​
    DATACAMP.COM
     and gTTS for speech synthesis​
    STACKOVERFLOW.COM
    , demonstrating how to combine these tools to create an engaging voice-enabled chatbot. With the provided code and guide, you should have a working chatbot that can be improved or expanded as needed, marking a solid starting point for more advanced AI assistants. Good luck with your chatbot development!
    """
        """
    CELL WITH INSTRUCTIONS
    To ensure I provide the most effective implementation details, could you clarify:

    What framework or environment are you using for the Marimo-based chatbot? Are you looking for a Python-based Marimo implementation or something else?
    Do you have any preferred text-to-speech (TTS) and speech-to-text (STT) libraries? For example, Google Speech API, Whisper, or pyttsx3 for voice synthesis?
    For the multilingual feature, should it be a simple button/toggle that switches between two predefined languages (e.g., English and Spanish), or should it support multiple language options?
    Should the chatbot support predefined responses, or do you want it to be AI-driven with an LLM backend (e.g., OpenAI's API)?
    Do you need the chatbot interface to be visually customized in any way, or is a simple functional implementation sufficient?
    Once I have these details, I’ll conduct the research and provide you with a full working implementation.






    You said:
    1. Marimo python. 2. Openai api. 3. Simple toggle. 4.openai api. 5. Customizable but at least the first version must work out of the box.
    ChatGPT said:
    Got it! I will research how to implement a simple chatbot using Marimo (Python) with OpenAI's API, including microphone input for speech recognition, text-to-speech functionality, and a simple toggle for switching between English and another language. The implementation will be functional out of the box but customizable for further improvements.

    I'll provide you with a full working code once the research is complete.

    Research completed in 5m · 19 sources
    Introduction
    Building a voice-interactive chatbot in Python is now easier with Marimo, an open-source reactive notebook platform. Marimo provides a ready-made chat UI component that can embed a custom chatbot interface directly into your notebook or app with just a few lines of code​
    SIMONWILLISON.NET
    ​
    SIMONWILLISON.NET
    . We’ll leverage this along with OpenAI’s API to create a simple chatbot that can handle both text and speech. The chatbot will support:
    Text and Voice Input: Users can type messages or press a microphone button and speak.
    AI Responses via OpenAI: The user’s input is sent to an OpenAI GPT model (e.g. GPT-3.5) to generate a reply.
    Text and Voice Output: The reply is displayed in the chat interface and simultaneously synthesized to speech so the user can listen.
    Language Toggle: A toggle switch lets users switch between English and another language (predefined) for both input and output, without complex internationalization.
    This guide will walk through the implementation step by step – from setting up the environment to integrating speech-to-text and text-to-speech – and culminate in a complete code example. Each section includes code snippets and explanations for seamless integration, ensuring the final solution works out of the box and is easy to customize.
    Prerequisites and Setup
    Before coding the chatbot, make sure you have the following:
    Python Environment: Ensure you have Python 3 installed. Marimo works within a Python environment (and even has an online playground​
    MARIMO.IO
    ), but you can also use it locally by installing the package.
    Marimo: Install Marimo, which provides the reactive notebook and UI components. You can install it via pip:
    bash
    Copy
    pip install marimo
    Marimo is a reactive Python notebook environment​
    MARIMO.IO
    , so you can run this code in a Marimo notebook or as a Marimo app.
    OpenAI API Key and SDK: Sign up for an OpenAI API key (from the OpenAI dashboard) and install the OpenAI Python library:
    bash
    Copy
    pip install openai
    This will let us call OpenAI’s GPT (for chat completion) and Whisper (for speech-to-text) endpoints.
    gTTS (Google Text-to-Speech): Install the gTTS library for text-to-speech conversion:
    bash
    Copy
    pip install gTTS
    gTTS uses Google’s Text-to-Speech to synthesize speech from text, which we’ll use to speak the chatbot’s responses.
    Note: In a Marimo notebook, you may also declare these as requirements. For simplicity, we assume you can use pip to get the needed packages. Once the environment is set, import the necessary libraries in your Python code (Marimo, openai, gTTS, etc.) and set your OpenAI API key (e.g. openai.api_key = "YOUR_API_KEY").
    Creating the Chat Interface with Marimo
    Marimo’s UI module (marimo.ui) provides a high-level chat element that handles the chat history and user input box for you​
    SIMONWILLISON.NET
    . You just need to provide a function (or model) that generates a response given the conversation so far. Marimo will:
    Display the conversation in a chat bubble format (user and assistant messages).
    Provide a text input box for the user to type new messages.
    Call your supplied function whenever a new user message is sent, passing in the message history, and then display the function’s return value as the assistant’s reply​
    SIMONWILLISON.NET
    .
    Using mo.ui.chat: We initialize the chat interface by calling mo.ui.chat(response_fn), where response_fn is our Python function to handle incoming messages and produce a reply. For example, a trivial echo chatbot in Marimo could be:
    python
    Copy
    import marimo as mo

    def echo_model(messages, config=None):
        # Simply echo back the last user message
        return f"Echo: {messages[-1].content}"

    chat_ui = mo.ui.chat(echo_model)
    chat_ui
    In this echo_model, messages is a list of ChatMessage objects (each has a .role like "user" or "assistant" and .content text​
    DOCS.MARIMO.IO
    ). We return a string, and Marimo will add that as an assistant message in the chat. This minimal example shows how easy it is to embed a chat in the UI​
    SIMONWILLISON.NET
    .For our chatbot, we want response_fn to call OpenAI’s API. Marimo actually provides built-in chat handlers for popular LLM providers (OpenAI, Anthropic, etc.) which you can use directly​
    SIMONWILLISON.NET
    . For instance, you could do:
    python
    Copy
    chat_ui = mo.ui.chat(
        mo.ai.llm.openai(model="gpt-3.5-turbo", api_key="YOUR_API_KEY", system_message="You are a helpful assistant.")
    )
    This uses Marimo’s built-in OpenAI connector (with a specified model and system prompt) to handle the chat responses​
    SIMONWILLISON.NET
    . However, to integrate custom features like our language toggle and audio output, we’ll write our own function using the OpenAI library directly. This gives us more control.Implementing the Response Function with OpenAI:
    Our response_fn will use OpenAI’s Chat Completion API (GPT-3.5 or GPT-4) to generate answers. Key steps in the function:
    Collect the conversation: We’ll take the messages list provided by Marimo (which contains the conversation history) and convert it into the format OpenAI’s API expects. This means a list of dicts like {"role": "user", "content": "..."} or {"role": "assistant", "content": "..."}. We can also inject a system message at the start to guide the assistant’s behavior (e.g. define its persona or language).
    Include language preference: If the user toggled the language switch (explained later), and for example chose Spanish, we can set the system message to “You are a helpful assistant. Respond in Spanish.” (Otherwise, it can be “Respond in English” or simply “You are a helpful assistant.”). This ensures the AI replies in the chosen language.
    Call OpenAI API: Use openai.ChatCompletion.create() with the model (e.g. "gpt-3.5-turbo"), your assembled message list, and get the assistant’s reply text.
    Return the reply: We’ll return the reply text so that Marimo displays it in the chat. But we actually want to return both text and an audio player for the reply, so in the next section we’ll modify this to include TTS output.
    Here’s a simplified version of the response function focusing on text:
    python
    Copy
    import openai

    openai.api_key = "YOUR_API_KEY"  # set your API key

    def chatbot_response(messages):
        # Determine system prompt based on language toggle (English/Spanish)
        if language_switch.value:  # Suppose True means Spanish mode
            system_msg = {"role": "system", "content": "You are a helpful assistant. Please respond in Spanish."}
        else:
            system_msg = {"role": "system", "content": "You are a helpful assistant. Respond in English."}
        # Build the message list for OpenAI (include system + conversation so far)
        conversation = [system_msg]
        for msg in messages:
            conversation.append({"role": msg.role, "content": msg.content})
        # Call OpenAI ChatCompletion
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=conversation)
        answer = response.choices[0].message.content
        return answer  # This will be displayed as the assistant's reply
    }

    chat_ui = mo.ui.chat(chatbot_response)
    In this snippet, language_switch is a UI element (to be defined later) that holds a boolean indicating the language choice. We prepend a system message accordingly. The OpenAI API call uses the entire conversation (including the latest user query) to generate a context-aware response. The returned answer string gets rendered as the assistant’s message in the chat UI.Note: The OpenAI ChatCompletion API expects conversation history for context. By including the prior messages (which Marimo maintains), the assistant can have a memory of the chat. The system message sets the default language and behavior. Also, ensure you handle the API key securely (you might use openai.api_key or an environment variable).
    Integrating Microphone Input (Speech-to-Text)
    To let users speak to the chatbot, we use Marimo’s Microphone UI element. The mo.ui.microphone() component renders a microphone button in the interface. When clicked, it will record audio from the user’s browser (after prompting for permission)​
    DOCS.MARIMO.IO
    . Once the recording is done, the audio data is available in Python as a binary blob (an io.BytesIO object)​
    DOCS.MARIMO.IO
    .Using mo.ui.microphone:
    We create a microphone element and specify an on_change callback. This callback function will be automatically invoked when a new audio recording is finished (i.e., when the microphone input value changes). The callback receives the audio data (as a BytesIO stream) as its argument. Our plan for the callback:
    Transcribe audio to text: Send the recorded audio to OpenAI’s Whisper API for transcription (speech-to-text). The OpenAI Python SDK provides openai.Audio.transcribe() to handle this using the Whisper model. We need to provide the model name (e.g., "whisper-1") and the audio file or file-like object. We’ll use the BytesIO from the mic directly.
    Handle audio format: Ensure the audio data is in a supported format and the API knows the format. Whisper API supports common formats like mp3, wav, m4a, webm, etc​
    DATACAMP.COM
    ​
    DATACAMP.COM
    . Marimo’s mic likely produces a WebM or WAV. To be safe, we can set the name attribute of the BytesIO object (e.g., audio_bytes.name = "voice.webm") so that the OpenAI SDK can infer the format​
    COMMUNITY.OPENAI.COM
    .
    Get the text result: The openai.Audio.transcribe() call returns the transcribed text (or a dictionary with the text). We extract the text string. For better accuracy, we can also specify the language to Whisper if we know it (OpenAI allows a language parameter​
    DATACAMP.COM
    ). For example, if the user selected Spanish mode, we might call openai.Audio.transcribe("whisper-1", audio_bytes, language="es"). Whisper can auto-detect language, but providing a hint can improve speed/accuracy​
    DATACAMP.COM
    .
    Submit as chat message: Take the transcribed text and inject it into the chat interface as if the user had typed it. Marimo’s chat element has a method send_message() that allows the backend to programmatically send a message into the chat UI​
    DOCS.MARIMO.IO
    . We’ll use chat_ui.send_message({"role": "user", "content": user_text}) to add the transcribed text as a user message. This will trigger our chatbot_response function to generate a reply (just like a normal user text input).
    Now, let’s implement the microphone and its callback:
    python
    Copy
    # Create a microphone UI element with a callback
    def transcribe_and_send(audio_bytes: io.BytesIO):
        # Ensure the audio BytesIO has a filename (extension) for format detection
        audio_bytes.name = "voice.webm"  # assuming webm/opus from browser; use "wav" if needed
        # Transcribe speech to text using OpenAI Whisper
        transcript = openai.Audio.transcribe(model="whisper-1", file=audio_bytes)
        # Extract the text (the API may return an object or dict with 'text')
        user_text = transcript.get('text') if isinstance(transcript, dict) else str(transcript)
        user_text = user_text.strip()
        if user_text:
            # Send this as a new user message in the chat UI
            chat_ui.send_message({"role": "user", "content": user_text})

    mic_button = mo.ui.microphone(label="🎤 Speak", on_change=transcribe_and_send)
    In this code:
    We label the microphone button with a microphone emoji and text for clarity.
    The transcribe_and_send function does the work of calling Whisper. We set audio_bytes.name so that the OpenAI SDK treats it as a file named “voice.webm” (WebM is a supported format​
    DATACAMP.COM
    ). Alternatively, if the mic yields WAV, use .name = "audio.wav". This little trick is needed because the OpenAI SDK infers the file type from the file name extension​
    COMMUNITY.OPENAI.COM
    .
    We call openai.Audio.transcribe("whisper-1", file=audio_bytes) and get the result. (If using an older version of openai SDK, the function might be openai.Audio.transcribe("whisper-1", audio_bytes) without explicit file= parameter.) We can also pass language="en" or "es" based on the toggle to guide the transcription​
    DATACAMP.COM
    .
    Once we have user_text, we use chat_ui.send_message() to inject it. The dict requires a "role" ("user" in this case) and the "content" string. Marimo will add this to the chat and call our chatbot_response function, resulting in the AI’s answer appearing.
    Now the user can click the mic button, speak, and the spoken words will be transcribed and appear as a chat message from the user, prompting the AI to respond. This happens almost as seamlessly as typing – the user doesn’t manually type the transcribed text; it’s done behind the scenes.
    Adding Text-to-Speech for Responses
    We want the chatbot’s replies not only in text, but also spoken aloud (or at least available as audio). To do this, we integrate a text-to-speech (TTS) step in the response generation. We’ll use gTTS (Google Text-to-Speech), a simple Python library that converts text to spoken audio.Using gTTS for TTS:
    gTTS can take a text string and a language code, and produce an audio file (MP3 by default) with the spoken version. Basic usage example from documentation:
    python
    Copy
    from gtts import gTTS
    text = "Hello, this is a text-to-speech example."
    tts = gTTS(text=text, lang='en', slow=False)
    tts.save("output.mp3")  # saves the speech audio to a file
    STACKOVERFLOW.COM
    . In our case, we don’t want to save to a file on disk; instead, we’ll use an in-memory file. gTTS provides a write_to_fp() method that writes the MP3 bytes to a file-like object​
    GTTS.READTHEDOCS.IO
    . We can create an io.BytesIO() buffer and write the audio data into it. This buffer can then be used to play the audio in the Marimo interface.Embedding Audio in Marimo:
    Marimo has an mo.audio() element that can take a source (URL or BytesIO) and render an HTML audio player​
    DOCS.MARIMO.IO
    . We will use mo.audio(bytes_buffer) to create an audio player for the chatbot’s reply. This player will appear in the chat bubble along with the text, so the user can immediately play the reply voice. (Depending on browser settings, the user might need to click play since auto-play of audio might be restricted. The audio element will at least provide the output without additional setup.)Integrating TTS in the Response Function:
    We modify our chatbot_response function to do the following when producing the assistant’s answer:
    After getting the answer text from OpenAI, generate speech audio from it. We choose the language for TTS based on the toggle (e.g., use 'en' for English or 'es' for Spanish so the accent/pronunciation is correct).
    Create a gTTS object with the answer text and language, and write it to a BytesIO buffer.
    Reset the buffer’s pointer to start (buffer.seek(0)) so it can be read from the beginning.
    Use mo.vstack([...]) to return a combined UI element containing both the answer text and an audio player. We stack a text element (we can use mo.md(answer_text) for nice formatting) above the audio player. Using a vertical stack ensures they appear together as one message​
    DOCS.MARIMO.IO
    .
    Here’s how the updated response function looks with TTS:
    python
    Copy
    import io
    from gtts import gTTS
    import marimo as mo

    def chatbot_response(messages):
        # Determine target language code from the toggle (False = English, True = Spanish)
        target_lang = "es" if language_switch.value else "en"
        system_msg = {"role": "system", "content": "You are a helpful assistant. Respond in Spanish." if target_lang=="es" 
                                                      else "You are a helpful assistant. Respond in English."}
        # Assemble conversation for OpenAI
        convo = [system_msg] + [{"role": m.role, "content": m.content} for m in messages]
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=convo)
        answer_text = response.choices[0].message.content.strip()
        # Text-to-speech conversion of the answer
        tts = gTTS(text=answer_text, lang=target_lang, slow=False)
        audio_data = io.BytesIO()
        tts.write_to_fp(audio_data)         # write MP3 data to buffer
        audio_data.seek(0)                  # reset buffer pointer
        # Return a combined element: Answer text + audio player
        return mo.vstack([ mo.md(answer_text), mo.audio(audio_data) ])
    Let’s break down the TTS part:
    We choose target_lang as "es" (Spanish) if the switch is on, otherwise "en" for English. gTTS supports many language codes (e.g., 'en', 'es', 'fr', etc.). The voice used will correspond to the language and a default accent (can be adjusted with a tld parameter if needed, e.g., tld="co.uk" for British English​
    GTTS.READTHEDOCS.IO
    , but we’ll keep it simple).
    gTTS(text=answer_text, lang=target_lang, slow=False) creates the TTS object. We use slow=False for normal speed.
    Instead of saving to a file, we do tts.write_to_fp(audio_data), where audio_data is a BytesIO buffer. This writes the MP3 bytes into audio_data​
    GTTS.READTHEDOCS.IO
    .
    We then ensure the buffer is at the beginning (audio_data.seek(0)).
    mo.audio(audio_data) creates an audio player element from this buffer. Marimo will handle converting it to an appropriate HTML audio tag in the output​
    DOCS.MARIMO.IO
    .
    We put the answer text (formatted via mo.md) and the audio element into a vertical stack container: mo.vstack([...]). According to Marimo’s docs, vstack will stack the given items vertically (one above the other)​
    DOCS.MARIMO.IO
    . This means in the chat bubble, the assistant’s response will show the text, and just below it, an audio bar to play the spoken answer. It will appear as a single assistant message containing both elements.
    Now our chat function returns a UI element (the stack), not just a string. Marimo’s chat is designed for this – it can handle messages that are text, images, plots, or other UI components as the response​
    SIMONWILLISON.NET
    . The user will see the text immediately and can click play to hear it.
    Implementing the Language Toggle
    The language toggle is a simple UI control to switch the interface between English and the second language (for example, Spanish). We don’t need a full i18n framework since we’re only dealing with two languages in a limited way. We will use a boolean switch that the user can flip.Using mo.ui.switch: Marimo offers a switch() component, which is essentially a toggle that holds a boolean value (False/True)​
    DOCS.MARIMO.IO
    . We can label this switch to indicate its function, e.g., “Spanish Mode” or “🇺🇸/🇪🇸” to show language. The switch value will be False for English (default) and True for Spanish in our design. We’ll refer to this in our code to decide on translation/transcription language and the assistant’s response language.We create the switch at the top-level so it’s always visible for the user to toggle. For example:
    python
    Copy
    language_switch = mo.ui.switch(label="Spanish Mode")
    By default this will render a toggle (perhaps a checkbox-style slider) labeled “Spanish Mode”. When unchecked (False), we treat it as English; when checked (True), we use Spanish. We already used language_switch.value inside our functions above to branch logic. Marimo’s reactive nature means whenever language_switch.value changes, it can trigger re-computation of dependent cells. In our case, the chat function will read its value each time it runs, so the next user query after toggling will use the updated language setting.What the toggle affects:
    OpenAI prompt: As shown, we change the system message to instruct the assistant to reply in the chosen language.
    Whisper transcription: We could pass the corresponding language code to Whisper for better accuracy (optional, since Whisper auto-detects).
    TTS voice: We pass the language code to gTTS so the spoken output uses the correct language/voice.
    Because we handle all these in code using the language_switch.value, the UI control itself is simple. We don’t need separate text labels for “English” vs “Spanish” because the system prompt and TTS handle it dynamically. The user just flips the switch to change the mode, and the next interactions will be in that language.
    Complete Code Example
    Below is the full code combining all the components – chat UI, OpenAI integration, microphone input, text-to-speech output, and language toggle. This code can be run in a Marimo notebook or app. When executed, it will display the chatbot interface with a toggle and a microphone button. Users can type or talk to the bot and get responses in text and audio.
    python
    Copy
    import marimo as mo  
    import openai, io  
    from gtts import gTTS  

    # Set your OpenAI API key  
    openai.api_key = "YOUR_OPENAI_API_KEY"  

    # Language toggle switch (False = English, True = Spanish)  
    language_switch = mo.ui.switch(label="Spanish Mode")  

    # Define the chatbot response function for mo.ui.chat  
    def chatbot_response(messages):  
        # Choose language based on toggle  
        use_spanish = language_switch.value  
        lang_code = "es" if use_spanish else "en"  
        # System prompt to enforce language  
        system_content = ( "You are a helpful assistant. Respond in Spanish."  
                           if use_spanish else  
                           "You are a helpful assistant. Respond in English." )  
        convo = [{"role": "system", "content": system_content}]  
        # Append conversation history (user and assistant messages so far)  
        for msg in messages:  
            convo.append({"role": msg.role, "content": msg.content})  
        # OpenAI ChatCompletion API call  
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=convo)  
        answer_text = response.choices[0].message.content.strip()  

        # Convert answer_text to speech audio  
        tts = gTTS(text=answer_text, lang=lang_code, slow=False)  
        audio_bytes = io.BytesIO()  
        tts.write_to_fp(audio_bytes)  
        audio_bytes.seek(0)  # reset pointer to beginning  

        # Return combined text + audio as the chat reply  
        return mo.vstack([  
            mo.md(answer_text),       # display answer text  
            mo.audio(audio_bytes)     # audio player for answer  
        ])  

    # Create the chat UI element with our response function  
    chat_ui = mo.ui.chat(chatbot_response)  

    # Microphone input with on_change callback for STT  
    def transcribe_and_send(audio_bytes):  
        # Name the BytesIO for format detection (assume webm audio)  
        audio_bytes.name = "speech.webm"  
        # Transcribe speech to text using OpenAI Whisper  
        transcript = openai.Audio.transcribe("whisper-1", file=audio_bytes)  
        text = transcript.get("text") if isinstance(transcript, dict) else str(transcript)  
        text = text.strip()  
        if text:  
            # Send the transcribed text as a user message to the chat  
            chat_ui.send_message({"role": "user", "content": text})  

    mic_button = mo.ui.microphone(label="🎤 Speak", on_change=transcribe_and_send)  

    # Layout: display the toggle, mic button, and chat UI in a vertical stack  
    mo.vstack([  
        mo.md("**Language:**"), language_switch,  
        mo.md("**Speak or Type your message**"), mic_button,  
        chat_ui  
    ])
    How it works: When you run this code, you’ll see a toggle switch for language, a “Speak” microphone button, and the chat interface. You can toggle the switch to your desired language at any time. To interact: either type a message in the chat input box and hit Enter, or click the microphone button and speak your question. If using the mic: after you stop recording, the audio will be sent to Whisper and the recognized text will appear as if you typed it. The OpenAI GPT model then generates a reply. The reply will appear in the chat with an audio player below it. You can read the text and click play to hear the spoken response.This fulfills the requirements: the user can converse by voice or text, and the assistant responds in both mediums. The code is modular and uses simple constructs, making it easy to extend. For example, you could swap out gTTS for another TTS service, or add more languages by replacing the switch with a dropdown (mo.ui.dropdown) of language options and adjusting the logic.
    Running and Customizing the Chatbot
    To run the chatbot, ensure you are in a Marimo environment (or have the Marimo app serving the UI). The interface elements (toggle, mic, chat) should render automatically from the code above. Here’s how a typical interaction would go:
    Select Language: Flip the Language switch to your desired language. (Off for English, on for Spanish in our example).
    Speak or Type: Click the 🎤 Speak button and record your question, or type a message into the chat input and send it. For instance, ask “What’s the weather today?” (in English mode) or “¿Cómo estará el clima hoy?” (in Spanish mode).
    Wait a moment: The audio (if used) is sent to OpenAI Whisper. Within a second or two, the transcribed text will appear in the chat as your message. The OpenAI model is then queried, which usually responds in a couple of seconds.
    Read and Listen to Reply: The chatbot’s answer appears as a chat bubble containing text and an audio player. For example, it might say in Spanish mode: “Según el pronóstico, hoy estará soleado con temperaturas agradables.” and an audio bar. Click the play button to hear it spoken aloud.
    Because we used streaming UI components, each step happens interactively. The design ensures simultaneous text and speech output – you don’t have to wait for the audio separately; it’s generated and available as soon as the text is displayed.Customization Options:
    Model and API: You can switch to a different OpenAI model (like gpt-4 if you have access) by changing the model name in ChatCompletion.create(). You could also integrate Anthropic’s Claude or other providers using Marimo’s built-in handlers​
    SIMONWILLISON.NET
    , but that would require their API keys and possibly different function calls.
    Language: Our toggle currently switches between English and Spanish. To change the second language, adjust the labels and language codes accordingly (e.g., to French 'fr' or German 'de'). For multiple languages, consider using a dropdown (mo.ui.dropdown or mo.ui.radio with options) instead of a simple switch.
    Speech Recognition: We used OpenAI’s Whisper API for transcription for simplicity and accuracy. If you want an offline solution, you might integrate a library like SpeechRecognition with a local engine, but that would require additional setup. Whisper can handle many languages and accents as shown in OpenAI’s documentation​
    DATACAMP.COM
    ​
    DATACAMP.COM
    .
    Text-to-Speech: gTTS was chosen for ease of use (no API key required and supports many languages). The trade-off is that it requires an internet connection to Google’s TTS service and returns an MP3. If you need offline TTS, you could use libraries like pyttsx3 (for a robotic voice) or use an API like Amazon Polly or Azure TTS for more natural voices. In Marimo, as long as you get an audio file or bytes, you can feed it to mo.audio().
    Finally, because Marimo is a reactive notebook, any changes to the code or UI state update the relevant parts automatically. This makes our chatbot quite responsive. We’ve avoided any complex internationalization frameworks – the code simply switches languages via conditionals. This straightforward approach is effective for a two-language scenario.References: The implementation draws on Marimo’s documentation and OpenAI’s API usage. Marimo’s chat UI and interactive elements (chat, microphone, switch, audio) enable the interface with minimal code​
    SIMONWILLISON.NET
    ​
    DOCS.MARIMO.IO
    . We used OpenAI’s Whisper for transcription​
    DATACAMP.COM
     and gTTS for speech synthesis​
    STACKOVERFLOW.COM
    , demonstrating how to combine these tools to create an engaging voice-enabled chatbot. With the provided code and guide, you should have a working chatbot that can be improved or expanded as needed, marking a solid starting point for more advanced AI assistants. Good luck with your chatbot development!
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    import openai, io

    from dotenv import load_dotenv
    import os

    load_dotenv()
    # Set your OpenAI API key
    openai.api_key = os.getenv('OPENAI_API_KEY')
    return io, load_dotenv, mo, openai, os


@app.cell
def _(gTTS, io, mo, openai):
    # Language toggle switch (False = English, True = Modo Português)
    language_switch = mo.ui.switch(label='Modo Português')


    # Define the chatbot response function for mo.ui.chat
    def chatbot_response(messages):
        # Choose language based on toggle
        use_portuguese = language_switch.value
        lang_code = 'pt' if use_portuguese else 'en'
        # System prompt to enforce language
        system_content = (
            'You are a helpful assistant. Respond in Spanish.'
            if use_portuguese
            else 'You are a helpful assistant. Respond in English.'
        )
        convo = [{'role': 'system', 'content': system_content}]
        # Append conversation history (user and assistant messages so far)
        for msg in messages:
            convo.append({'role': msg.role, 'content': msg.content})
        # OpenAI ChatCompletion API call
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo', messages=convo
        )
        answer_text = response.choices[0].message.content.strip()

        # Convert answer_text to speech audio
        tts = gTTS(text=answer_text, lang=lang_code, slow=False)
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)  # reset pointer to beginning

        # Return combined text + audio as the chat reply
        return mo.vstack(
            [
                mo.md(answer_text),  # display answer text
                mo.audio(audio_bytes),  # audio player for answer
            ]
        )
    return chatbot_response, language_switch


@app.cell
def _():
    '''
    import modal

    stub = modal.App()

    @stub.function(secrets=[modal.Secret.from_name("openai-api-key")])

    async def transcribe_and_send(audio_bytes: bytes, chat_ui: modal.Dict):
        """Transcribes audio bytes to text using OpenAI's Whisper API and sends the text to the chat UI."""

        # Convert bytes to a BytesIO object
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "speech.webm"  # Important for format detection

        try:
            transcript = await openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"  # Directly get the text
            )

            text = transcript.strip()
            if text:
                # Send the transcribed text as a user message to the chat
                await chat_ui.put({"role": "user", "content": text})

        except Exception as e:
            print(f"Error during transcription: {e}")
            await chat_ui.put({"role": "assistant", "content": f"Error transcribing audio: {e}"})  # Send error to chat


    @stub.local_entrypoint()
    def main():
        import modal.ui
        import asyncio

        # Initialize a dictionary to hold chat messages.  This is crucial for Modal's UI.
        chat_messages = modal.Dict.ephemeral()

        async def chatbot_response(message, chat_history):
            """Placeholder for your actual chatbot logic.  Replace with your desired LLM interaction."""
            await asyncio.sleep(1)  # Simulate a delay
            return "This is a dummy response.  Replace me with your LLM's output."

        # Create the chat UI element with our response function
        chat_ui = modal.ui.chat(chatbot_response, chat_messages=chat_messages)  # Pass the chat_messages dict

        # Microphone input with on_change callback for STT
        async def mic_callback(audio_bytes: bytes):
            """Callback function for the microphone.  It calls the Modal function to transcribe and send."""
            await transcribe_and_send.call(audio_bytes, chat_messages)  # Pass the chat_messages dict

        mic_button = modal.ui.microphone(label="🎤 Speak", on_change=mic_callback)

        # Language switch (example, replace with your actual implementation)
        language_switch = modal.ui.checkbox(label="English", value=True)  # Example

        # Layout: display the toggle, mic button, and chat UI in a vertical stack
        modal.ui.vstack([
            modal.ui.md("**Language:**"), language_switch,
            modal.ui.md("**Speak or Type your message**"), mic_button,
            chat_ui
        ])

    '''
    return


@app.cell
def _(chat_ui, gTTS, io, mo, openai):
    import websocket
    import json
    import threading
    import queue


    class ChatbotStream:
        def __init__(self):
            self.ws = None
            self.message_queue = queue.Queue()
            self.is_connected = False

        def connect(self):
            websocket.enableTrace(True)
            self.ws = websocket.WebSocketApp(
                'wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17',
                header=[
                    'Authorization: Bearer ' + openai.api_key,
                    'OpenAI-Beta: realtime=v1',
                ],
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close,
                on_open=self.on_open,
            )

            wst = threading.Thread(target=self.ws.run_forever)
            wst.daemon = True
            wst.start()

        def on_message(self, ws, message):
            data = json.loads(message)
            self.message_queue.put(data)

        def on_error(self, ws, error):
            print(f'Error: {error}')

        def on_close(self, ws, close_status_code, close_msg):
            print('Connection closed')
            self.is_connected = False

        def on_open(self, ws):
            print('Connected to OpenAI Realtime API')
            self.is_connected = True

        def send_message(self, text):
            if not self.is_connected:
                return

            event = {'type': 'message', 'content': text}
            self.ws.send(json.dumps(event))

        def get_response(self):
            try:
                return self.message_queue.get_nowait()
            except queue.Empty:
                return None


    def process_stream():
        chatbot = ChatbotStream()
        chatbot.connect()

        while True:
            response = chatbot.get_response()
            if response:
                if response.get('type') == 'text':
                    text = response['content']
                    tts = gTTS(text=text, lang='en')
                    audio_bytes = io.BytesIO()
                    tts.write_to_fp(audio_bytes)
                    audio_bytes.seek(0)

                    chat_ui.send_message(
                        {
                            'role': 'assistant',
                            'content': mo.vstack(
                                [mo.md(text), mo.audio(audio_bytes)]
                            ),
                        }
                    )


    stream_thread = threading.Thread(target=process_stream)
    stream_thread.daemon = True
    stream_thread.start()
    return (
        ChatbotStream,
        json,
        process_stream,
        queue,
        stream_thread,
        threading,
        websocket,
    )


@app.cell
def _(chatbot, mo, queue, threading, websocket):
    import sounddevice as sd
    import numpy as np


    # Audio configuration
    SAMPLE_RATE = 16000
    CHANNELS = 1
    DTYPE = np.int16
    CHUNK_DURATION = 0.1  # seconds

    # Create audio queue
    audio_queue = queue.Queue()


    def audio_callback(indata, frames, time, status):
        if status:
            print(f'Error: {status}')
        audio_queue.put(indata.copy())


    def start_recording():
        try:
            stream = sd.InputStream(
                channels=CHANNELS,
                samplerate=SAMPLE_RATE,
                dtype=DTYPE,
                callback=audio_callback,
                blocksize=int(SAMPLE_RATE * CHUNK_DURATION),
            )
            stream.start()
            return stream
        except Exception as e:
            print(f'Error starting audio stream: {e}')
            return None


    def stop_recording(stream):
        if stream:
            stream.stop()
            stream.close()


    def process_audio():
        stream = None
        try:
            stream = start_recording()
            while True:
                audio_chunk = audio_queue.get()
                if audio_chunk is None:
                    break

                # Convert to bytes and send to websocket
                audio_bytes = audio_chunk.tobytes()
                chatbot.ws.send(audio_bytes, opcode=websocket.OPCODE_BINARY)

        except Exception as e:
            print(f'Error processing audio: {e}')
        finally:
            if stream:
                stop_recording(stream)


    # Start audio processing in a separate thread
    audio_thread = threading.Thread(target=process_audio)
    audio_thread.daemon = True
    audio_thread.start()

    # Create microphone button
    mic_active = mo.state(False)


    def toggle_mic():
        mic_active.value = not mic_active.value
        if mic_active.value:
            audio_queue.queue.clear()  # Clear any pending audio
        else:
            audio_queue.put(None)  # Signal to stop processing


    return (
        CHANNELS,
        CHUNK_DURATION,
        DTYPE,
        SAMPLE_RATE,
        audio_callback,
        audio_queue,
        audio_thread,
        mic_active,
        np,
        process_audio,
        sd,
        start_recording,
        stop_recording,
        toggle_mic,
    )


@app.cell
def _(mo, toggle_mic):
    mic_button = mo.ui.button('🎤', toggle_mic)
    return (mic_button,)


if __name__ == "__main__":
    app.run()
