import streamlit as st
from STT1 import AudioTranscriber
from MT import IndicTranslator
from gtts import gTTS
import os
import tempfile

# Define available languages and their codes
indian_languages = {
    'asm_Beng': 'as', 'ben_Beng': 'bn', 'brx_Deva': 'brx', 'doi_Deva': 'doi', 'gom_Deva': 'gom', 'guj_Gujr': 'gu',
    'hin_Deva': 'hi', 'kan_Knda': 'kn', 'kas_Arab': 'ks', 'kas_Deva': 'ks', 'mai_Deva': 'mai', 'tel_Telu': 'te',
    'tam_Taml': 'ta'
}

lang_dict = {"Assamese" : "asm_Beng","Bengali" : "ben_Beng","Sanskrit" : "san_Deva","Malayalam" : "mal_Mlym","English" : "eng_Latn",
"Marathi" : "mar_Deva","Tamil" : "tam_Taml","Manipuri" : "Meitei","Telugu" : "tel_Telu","Hindi" : "hin_Deva","Nepali" : "npi_Deva",
"Urdu" : "urd_Arab","Kannada" : "kan_Knda"}

# Initialize classes
transcriber = AudioTranscriber()
translator = IndicTranslator()

# Streamlit app
st.title("Real Time Indian Language Translation")
st.write("Translate spoken language seamlessly between Indian languages.")

# Step 1: Upload or record audio
st.header("Step 1: Upload or Record Audio")
audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

# Placeholder for playing recorded audio
if audio_file:
    st.audio(audio_file, format='audio/wav', start_time=0)

# Step 2: Transcribe audio
if audio_file and st.button("Transcribe Audio"):
    with st.spinner("Transcribing audio..."):
        try:
            temp_audio = tempfile.NamedTemporaryFile(delete=False)
            temp_audio.write(audio_file.read())
            temp_audio.close()

            transcription = transcriber.transcribe(temp_audio.name)
            st.success("Transcription completed!")
            st.text(f"Transcribed Text: {transcription}")
        except Exception as e:
            st.error(f"Transcription failed: {str(e)}")

# Step 3: Translate text
if 'transcription' in locals():
    st.header("Step 3: Translate Text")
    source_lang = st.selectbox("Select Source Language", list(lang_dict.keys()), index=9)  # Default: Kannada
    target_lang = st.selectbox("Select Target Language", list(lang_dict.keys()), index=12)  # Default: Telugu

    if st.button("Translate Text"):
        with st.spinner("Translating text..."):
            try:
                translations = translator.translate(
                    [transcription],
                    lang_dict[source_lang],
                    lang_dict[target_lang]
                )
                translated_text = translations[0]
                st.success("Translation completed!")
                st.text(f"Translated Text: {translated_text}")
            except Exception as e:
                st.error(f"Translation failed: {str(e)}")

# Step 4: Text-to-Speech
if 'translated_text' in locals():
    st.header("Step 4: Text-to-Speech")
    if st.button("Convert to Speech"):
        with st.spinner("Converting text to speech..."):
            try:
                language_code = indian_languages[lang_dict[target_lang]]
                speech = gTTS(text=translated_text, lang=language_code, slow=False)
                output_path = f"output_{language_code}.mp3"
                speech.save(output_path)
                st.audio(output_path, format="audio/mp3")
                st.success("Speech conversion completed!")
            except Exception as e:
                st.error(f"Speech conversion failed: {str(e)}")

# Step 5: Display animations and responses
st.balloons()
st.write("🎉 All steps completed! Enjoy your translated speech.")
