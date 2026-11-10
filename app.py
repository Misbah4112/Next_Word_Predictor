import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

BASE = Path(__file__).resolve().parent
st.set_page_config(page_title='NextWord AI | Neural Text Studio', page_icon='✦', layout='wide', initial_sidebar_state='expanded')

st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root {color-scheme:dark}
html,body,[class*="css"],.stApp {font-family:'DM Sans',sans-serif}
.stApp {background:radial-gradient(ellipse at 70% -15%,#24204c 0%,transparent 45%),#0a0d18;color:#edf0ff}
.block-container {padding-top:2rem;max-width:1200px}
[data-testid="stSidebar"] {background:#101425;border-right:1px solid #242a43}
h1,h2,h3 {font-family:'Space Grotesk',sans-serif!important;letter-spacing:-.035em}
h1 {font-size:3.1rem!important}
.hero {padding:35px 38px;border:1px solid #303557;border-radius:25px;background:linear-gradient(115deg,#1c2041,#12182a 65%,#132b38);margin-bottom:26px;position:relative}
.eyebrow {color:#9ba8ff;font-size:.78rem;font-weight:700;letter-spacing:.16em;text-transform:uppercase}
.hero-title {font-family:'Space Grotesk',sans-serif;font-size:3.3rem;font-weight:700;line-height:1.12;margin:14px 0;color:#f6f6ff}
.gradient {background:linear-gradient(90deg,#b7a5ff,#75e3ed);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.hero-sub {color:#abb4d0;max-width:620px;font-size:1.02rem;line-height:1.7}
.panel {border:1px solid #2b324e;background:#141a2c;border-radius:18px;padding:20px 23px;margin-bottom:15px}
.pill {display:inline-block;border-radius:100px;background:#27294b;border:1px solid #494775;color:#c9c5ff;padding:6px 12px;font-size:.75rem;margin-right:7px;margin-top:10px}
.section-kicker {color:#aab3d6;font-size:.82rem;text-transform:uppercase;letter-spacing:.11em;font-weight:700}
[data-testid="stMetric"] {background:#151c30;border:1px solid #2a324d;border-radius:16px;padding:16px 18px}
[data-testid="stMetricLabel"] {color:#9ba6c5}
[data-testid="stMetricValue"] {color:#f0f2ff}
.stButton>button[kind="primary"] {background:linear-gradient(90deg,#7768f7,#4d9cd6);border:0;color:white;border-radius:12px;font-weight:700;min-height:48px}
.stButton>button {border-radius:12px;min-height:44px;border-color:#39405f}
[data-testid="stTextInput"] input {background:#171d31;border:1px solid #3c4260;color:#f5f6ff;border-radius:12px;padding:14px}
[data-testid="stTabs"] button {font-weight:700}
hr {border-color:#2b324b!important}
.footer {text-align:center;color:#7783a6;padding:24px 0;font-size:.8rem}
</style>''', unsafe_allow_html=True)

@st.cache_resource(show_spinner='Loading neural networks…')
def load_resources():
    required = ['lstm_model.keras', 'gru_model.keras', 'tokenizer.pkl', 'config.json']
    missing = [f for f in required if not (BASE / f).exists()]
    if missing:
        raise FileNotFoundError('Missing files: ' + ', '.join(missing))
    models = {
        'LSTM': tf.keras.models.load_model(BASE / 'lstm_model.keras', compile=False),
        'GRU': tf.keras.models.load_model(BASE / 'gru_model.keras', compile=False),
    }
    # Only load pickle files that you created yourself.
    with (BASE / 'tokenizer.pkl').open('rb') as f:
        tokenizer = pickle.load(f)
    with (BASE / 'config.json').open('r') as f:
        config = json.load(f)
    max_len = int(config['max_sequence_len'])
    if max_len < 2:
        raise ValueError('max_sequence_len must be at least 2')
    for name, model in models.items():
        expected = model.input_shape[-1]
        if expected is not None and expected != max_len - 1:
            raise ValueError(f'{name} expects {expected} tokens, but config implies {max_len - 1}.')
    return models, tokenizer, max_len

try:
    models, tokenizer, max_len = load_resources()
except Exception as exc:
    st.error(f'Unable to load your saved training artifacts: {exc}')
    st.info('Place lstm_model.keras, gru_model.keras, tokenizer.pkl and config.json next to app.py. Run your training script to create them if needed.')
    st.stop()

index_to_word = {idx: word for word, idx in tokenizer.word_index.items()}
vocab_size = len(tokenizer.word_index) + 1

def predict(text, model, k):
    # Match training preprocessing (lowercase and remove punctuation).
    cleaned = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    cleaned = ' '.join(cleaned.split())
    if not cleaned:
        return []
    tokens = tokenizer.texts_to_sequences([cleaned])[0]
    oov_id = tokenizer.word_index.get(tokenizer.oov_token)

    if not tokens or all(t == oov_id for t in tokens):
        return []
    padded = pad_sequences([tokens], maxlen=max_len - 1, padding='pre', truncating='pre')
    scores = model.predict(padded, verbose=0)[0]
    ranked = np.argsort(scores)[::-1]
    result = []
    for idx in ranked:
        word = index_to_word.get(int(idx))
        if word and word.lower() != '<oov>':
            result.append((word, float(scores[idx])))
            if len(result) >= k:
                break
    return result

def generate(text, model, n):
    current = text.strip()
    for _ in range(n):
        choices = predict(current, model, 1)
        if not choices:
            break
        current += ' ' + choices[0][0]
    return current

def chart(predictions, title):
    labels = [w for w, _ in predictions][::-1]
    probs = [p * 100 for _, p in predictions][::-1]
    fig = go.Figure(go.Bar(x=probs, y=labels, orientation='h', marker=dict(color=probs, colorscale=[[0,'#6657c8'],[1,'#65d4ed']]), text=[f'{p:.2f}%' for p in probs], textposition='outside', cliponaxis=False))
    fig.update_layout(title=dict(text=title,font=dict(size=17,color='#edf0ff')), paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font=dict(color='#c1c9e2'),height=max(290,75+len(labels)*49),margin=dict(l=10,r=65,t=50,b=35),xaxis=dict(title='Predicted probability (%)',gridcolor='#303750',zeroline=False),yaxis=dict(title='',gridcolor='rgba(0,0,0,0)'))
    return fig

with st.sidebar:
    st.markdown('### ✦ NEXTWORD AI')
    st.caption('NEURAL TEXT STUDIO · v1.0')
    st.divider()
    st.markdown('#### Model controls')
    selected = st.radio('Neural architecture', ['LSTM','GRU'], horizontal=True)
    top_k = st.slider('Predictions to show',1,10,5)
    n_words = st.slider('Continuation length',1,30,8)
    st.divider()
    st.markdown('#### Model information')
    st.caption('Dataset: English quotes · Architecture: Embedding → RNN → Softmax')
    st.caption(f'Vocabulary: {vocab_size:,} tokens')
    st.caption(f'Maximum input context: {max_len - 1} tokens')
    st.success('● Models loaded successfully')

st.markdown('''<div class="hero"><div class="eyebrow">✦ Neural language playground</div><div class="hero-title">Predict the <span class="gradient">next word.</span><br>Explore the possibilities.</div><div class="hero-sub">Experience neural text prediction with two recurrent architectures. Explore word probabilities, generate continuations, and compare how LSTM and GRU think.</div><span class="pill">✧ LSTM</span><span class="pill">✧ GRU</span><span class="pill">✧ Live inference</span></div>''',unsafe_allow_html=True)

c1,c2,c3 = st.columns(3)
with c1: st.metric('Active model',selected)
with c2: st.metric('Vocabulary size',f'{vocab_size:,}')
with c3: st.metric('Context window',f'{max_len - 1} tokens')

st.write('')
st.markdown('### Your writing canvas')
st.caption('Enter a phrase to discover the words your model expects to come next.')
seed = st.text_input('Start typing',placeholder='Try: life is full of',label_visibility='collapsed')
if not seed.strip():
    st.caption('💡 Try a short phrase such as “life is”, “the world”, or “love is”.')

predict_tab, generate_tab, compare_tab = st.tabs(['🔮  Predict','✨  Generate','⚖️  Compare models'])

with predict_tab:
    st.write('')
    if st.button('Analyze next words',type='primary',use_container_width=True,key='predict'):
        if not seed.strip():
            st.warning('Enter some text above to get started.')
        else:
            with st.spinner('Analyzing your sentence…'):
                st.session_state['pred_results'] = (seed,selected,top_k,predict(seed,models[selected],top_k))
    saved = st.session_state.get('pred_results')
    if saved:
        old_seed,old_model,old_k,results = saved
        st.caption(f'Results for “{old_seed}” · {old_model} · Top {old_k}')
        if results:
            st.plotly_chart(chart(results,'Top next-word candidates'),use_container_width=True)
            df = pd.DataFrame(results,columns=['Word','Probability'])
            df['Probability (%)'] = (df['Probability']*100).round(3)
            st.dataframe(df[['Word','Probability (%)']],hide_index=True,use_container_width=True)
            st.download_button('↓ Export predictions (CSV)',df.to_csv(index=False).encode('utf-8'),file_name='nextword_predictions.csv',mime='text/csv')
        else:
            st.warning('No known words were recognized. Try another phrase using common English words.')
    else:
        st.info('Your probability chart will appear here after you analyze a phrase.')

with generate_tab:
    st.write('')
    st.markdown('#### Continue the thought')
    st.caption('Generates one word at a time by repeatedly selecting the most probable next word.')
    if st.button('Generate continuation',type='primary',use_container_width=True,key='generate'):
        if not seed.strip():
            st.warning('Enter a starting phrase above.')
        else:
            with st.spinner('Writing with your neural network…'):
                st.session_state['generated'] = (selected,generate(seed,models[selected],n_words))
    if 'generated' in st.session_state:
        which,generated = st.session_state['generated']
        st.markdown(f'<div class="panel"><div class="section-kicker">{which} generated text</div><h3 style="line-height:1.6;margin-top:12px">{__import__("html").escape(generated)}</h3></div>',unsafe_allow_html=True)
        st.download_button('↓ Download text',generated,file_name='generated_text.txt',mime='text/plain')

with compare_tab:
    st.write('')
    st.markdown('#### Two architectures. One prompt.')
    st.caption('Compare next-word probabilities and generated continuations from both trained models.')
    if st.button('Run side-by-side comparison',type='primary',use_container_width=True,key='compare'):
        if not seed.strip():
            st.warning('Enter a phrase above to compare models.')
        else:
            with st.spinner('Comparing neural architectures…'):
                st.session_state['comparison'] = (seed,top_k,n_words,{name:{'pred':predict(seed,model,top_k),'text':generate(seed,model,n_words)} for name,model in models.items()})
    if 'comparison' in st.session_state:
        comp_seed,comp_k,comp_n,comparison = st.session_state['comparison']
        st.caption(f'Comparison for “{comp_seed}” · Top {comp_k} · {comp_n} generated words')
        left,right=st.columns(2)
        for col,name in [(left,'LSTM'),(right,'GRU')]:
            with col:
                st.markdown(f'### {name}')
                st.markdown(f'<div class="panel">{__import__("html").escape(comparison[name]["text"])}</div>',unsafe_allow_html=True)
                if comparison[name]['pred']:
                    st.plotly_chart(chart(comparison[name]['pred'],f'{name} next-word predictions'),use_container_width=True,key=f'chart_{name}')
                else:
                    st.warning('No known words recognized.')

st.markdown('<div class="footer">NEXTWORD AI · Built with TensorFlow, Streamlit & Plotly · Predictions reflect the training quotes, not a general-purpose language model.</div>',unsafe_allow_html=True)
