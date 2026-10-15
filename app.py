import streamlit as st
import pandas as pd
from scipy.sparse import hstack
import urllib.parse
import urllib.request
import json
import warnings
from streamlit_mic_recorder import speech_to_text
from transformers import pipeline


warnings.filterwarnings('ignore')

# 1. Page Configuration
st.set_page_config(
    page_title="Disaster Tweet Classifier", 
    page_icon="🚨", 
    layout="centered"
)

# Robust Translation Function using Google Public Web API
def translate_text(text):
    if not text.strip():
        return text
    try: 
        encoded_text = urllib.parse.quote(text)
        url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=en&dt=t&q={encoded_text}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req)
        data = json.loads(response.read().decode('utf-8'))
        
        # Combine translated segments
        translated_segments = [item[0] for item in data[0] if item[0]]
        return "".join(translated_segments)
    except Exception as e:
        return text

# 2. Custom Styling
st.markdown("""
<style>
    /* Main App Background & Text Color */
    .stApp {
        background-color: #0b0f19 !important;
        color: #ffffff !important;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #111827 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Main Title Styling */
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #ffb44b;
        text-align: center;
        margin-bottom: 5px;
    }
    
    /* Sub Title Styling */
    .sub-title {
        font-size: 1.05rem;
        color: #94a3b8;
        text-align: center;
        margin-bottom: 25px;
    }
    
    /* Text Area Styling */
    .stTextArea textarea {
        background-color: #1e293b !important;
        color: #ffffff !important;
        border-radius: 10px;
        border: 1px solid #334155;
        font-size: 1rem;
    }
    
    /* Text Area Focus Effect */
    .stTextArea textarea:focus {
        border-color: #ffb44b;
        box-shadow: 0 0 8px rgba(255, 180, 75, 0.3);
    }

    /* Buttons Styling */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# 3. Sidebar
with st.sidebar:
    st.header("ℹ️ About the App")
    st.write("This is an AI application that detects real-time emergency feeds.")
    
    st.markdown("---")
    st.subheader("🌐 Supported Languages")
    st.write("• English\n\n• Hindi (हिंदी)\n\n• Gujarati (ગુજરાતી)")

    st.markdown("---")
        # Developer Credit Footer with LinkedIn Link
    st.markdown("### 👩‍💻 Developed By")
    st.markdown("*Ankita Vaghela* | Data Science & AI")
    st.markdown("[🔗 Connect on LinkedIn](https://www.linkedin.com/in/ankita-vaghela-018498392)")

    st.markdown("---")
    st.subheader("💡 Quick Tip")
    st.caption("Try pasting real emergency alerts, news headlines, or daily social media updates to test accuracy.")

    

# 4. Header Section
st.markdown('<div class="main-title">🚨 Real-Time Disaster Tweet Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Detect emergency & disaster alerts instantly from social media posts</div>', unsafe_allow_html=True)

# 5. Load Hugging Face BERT Model Pipeline
@st.cache_resource
def load_bert_model():
    classifier = pipeline("text-classification", model="elam2909/bert-disaster-classifier", return_all_scores=True)
    return classifier

try:
    with st.spinner("Loading BERT Transformer Model... Please wait."):
        classifier = load_bert_model()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")

# # 6. User Input Section with Voice (Mic) Support
st.subheader("📝 Enter Message / Tweet")


# Select Language for Speech Recognition
selected_lang = st.selectbox(
        "🎙️ Choose Voice Input Language:",
        options=["gu-IN", "hi-IN", "en-US"],
        format_func=lambda x: "Gujarati (ગુજરાતી)" if x == "gu-IN" else ("Hindi (हिंदी)" if x == "hi-IN" else "English"),
        key="voice_lang_select"
    )

    # Voice input using mic recorder with dynamic language support
spoken_text = speech_to_text(
        language=selected_lang, 
        start_prompt="🎙️ Click to Speak", 
        stop_prompt="⏹️ Stop Recording", 
        just_once=False, 
        key='mic_input'
    )

# If user speaks, use that text, otherwise allow manual typing
default_text = spoken_text if spoken_text else ""

user_tweet = st.text_area(
    "Paste text in any language or use the mic above:",
    value=default_text,
    height=100,
    placeholder="e.g., A massive earthquake has been reported near the coast. Emergency teams..."
)

# 7. Classification Logic
if st.button("Classify Tweet 🔍", use_container_width=True, type="primary"):
    if not user_tweet.strip():
        st.warning("Please enter some text to classify.")
    else:
        with st.spinner("Translating & Analyzing with BERT AI..."):
            # Step A: Direct Google API Translation
            translated_tweet = translate_text(user_tweet)
            
            # Show translated text in blue info box
            st.info(f"🌐 Translated Text for AI: \"{translated_tweet}\"")
            
            # Step B: Predict using Hugging Face BERT Pipeline
            results = classifier(translated_tweet)
            
            # Handle different pipeline output structures safely
            if isinstance(results, list) and len(results) > 0:
                if isinstance(results[0], list):
                    scores_list = results[0]
                else:
                    scores_list = results
            else:
                scores_list = []

            

            # Step C: Find the label with highest score
            if scores_list:
                best_prediction = max(scores_list, key=lambda x: x['score'])
                label = best_prediction['label']
                confidence = best_prediction['score'] * 100
            else:
                label = "not_disaster"
                confidence = 0.0
            
            # Step D: Map label safely (Handling 'not_disaster' vs 'disaster')
            label_lower = label.lower()
            if 'not' in label_lower or '0' in label_lower or 'non' in label_lower:
                is_disaster = False
            else:
                is_disaster = True

        st.divider()
        
        # Display Final Prediction
        if is_disaster:
            st.error("🚨 *DISASTER EVENT DETECTED*")
            st.write("⚠️ This tweet indicates a critical emergency, natural calamity, or real-world disaster report.")
        else:
            st.success("✅ *NON-DISASTER TWEET*")
            st.write("🟢 This tweet represents routine activity, general news, or non-emergency context.")
            
        # Confidence Metric Score & Progress Bar
        st.metric(label="BERT Confidence Score", value=f"{confidence:.2f}%")
        st.progress(int(confidence))
        