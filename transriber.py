from io import BytesIO
import streamlit as st
import whisper


@st.cache
def transcribe(file: BytesIO):
    # save to audios dir with unique name
    filename = f"audios/{file.name}"
    with open(filename, 'wb') as f:
        f.write(file.getbuffer())

    model = whisper.load_model("small")
    result = model.transcribe(filename)
	
    # # load audio and pad/trim it to fit 30 seconds
    # audio = whisper.load_audio(filename)
    # audio = whisper.pad_or_trim(audio)

    # # make log-Mel spectrogram and move to the same device as the model
    # mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # # detect the spoken language
    # # _, probs = model.detect_language(mel)
    # # print(f"Detected language: {max(probs, key=probs.get)}")

    # # decode the audio
    # options = whisper.DecodingOptions(without_timestamps=True, fp16=False)
    # result = model.decode(mel, options)

    # # print the recognized text
    # return result.text
    return result["text"]


st.title('Transcriber')

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    text = transcribe(uploaded_file)
    st.write(text)
