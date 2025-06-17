import streamlit as st
from transformers import pipeline, FlaxAutoModelForSequenceClassification, AutoTokenizer
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Cyberbullying Detection AI",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Cyberbullying Detection AI")
st.markdown("""
This AI system analyzes text for toxic content and cyberbullying.
Enter your text below to check for potentially harmful content.
""")

# Model loading with fallback to Flax
@st.cache_resource
def load_model():
    try:
        with st.spinner("Loading AI model... This may take a few moments."):
            # Attempting PyTorch
            classifier = pipeline(
                task="text-classification",
                model="Hate-speech-CNERG/bert-base-uncased-hatexplain",
                return_all_scores=True
            )
            return classifier
    except Exception as e:
        st.warning(f"PyTorch loading failed. Attempting Flax model. Error: {str(e)}")
        try:
            flax_model = FlaxAutoModelForSequenceClassification.from_pretrained(
                "Hate-speech-CNERG/bert-base-uncased-hatexplain", from_flax=True
            )
            tokenizer = AutoTokenizer.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain")
            return flax_model, tokenizer
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None

# Input text
text_input = st.text_area("Enter text to analyze:", height=150)

# Analyze button
if st.button("Analyze Text"):
    if text_input.strip():
        model_data = load_model()
        
        if model_data is not None:
            try:
                if isinstance(model_data, tuple):
                    flax_model, tokenizer = model_data
                    inputs = tokenizer(text_input, return_tensors="np")
                    outputs = flax_model(**inputs)
                    st.write("Flax Model Output:", outputs)
                else:
                    with st.spinner("Analyzing text using PyTorch..."):
                        results = model_data(text_input)[0]
                        df = pd.DataFrame(results)
                        
                        st.subheader("Analysis Results")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            for result in results:
                                label = result['label']
                                score = result['score']
                                color = "red" if "hate" in label.lower() or "offensive" in label.lower() else "green"
                                st.markdown(f"**{label}:**")
                                st.progress(score)
                                st.markdown(f"Score: {score:.2%}")
                                st.markdown("---")
                        
                        with col2:
                            hate_scores = [r for r in results if "hate" in r['label'].lower()]
                            max_hate = max(hate_scores, key=lambda x: x['score']) if hate_scores else None
                            
                            if max_hate and max_hate['score'] > 0.7:
                                st.error("⚠️ Harmful content detected!")
                            elif max_hate and max_hate['score'] > 0.4:
                                st.warning("⚠️ Potentially inappropriate content")
                            else:
                                st.success("✅ Safe content detected")
                            
                            st.markdown("### Detailed Analysis")
                            st.dataframe(df.sort_values(by='score', ascending=False))
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
    else:
        st.warning("Please enter some text to analyze.")

# Footer
st.markdown("---")
st.markdown("""
### About
This system uses AI models to detect various forms of harmful content:
- Hate Speech
- Offensive Language
- Normal Content
The model provides confidence scores to help identify harmful content.
""")
