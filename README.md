import streamlit as st
from transformers import pipeline
import tempfile
import os
import whisper

st.title("تطبيق نسخ وتلخيص الاجتماعات")
st.write("سجل اجتماعك أو ارفع ملف صوتي للحصول على النص والملخص.")

# رفع ملف صوتي
audio_file = st.file_uploader("اختر ملف صوتي (MP3/WAV)", type=["mp3", "wav"])

if audio_file is not None:
    st.audio(audio_file, format="audio/wav")
    
    # حفظ الملف مؤقتًا
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(audio_file.read())
    
    # تحويل الصوت إلى نص باستخدام Whisper
    model = whisper.load_model("base")
    result = model.transcribe(tfile.name)
    full_text = result["text"]
    
    st.subheader("النص الكامل:")
    st.write(full_text)
    
    # تلخيص النص باستخدام Hugging Face
    summarizer = pipeline("summarization")
    summary = summarizer(full_text, max_length=150, min_length=50, do_sample=False)
    
    st.subheader("الملخص:")
    st.write(summary[0]['summary_text'])
    
    # حذف الملف المؤقت
    os.unlink(tfile.name)
