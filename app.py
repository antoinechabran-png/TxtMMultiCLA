import streamlit as st
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
import networkx as nx
from community import community_louvain
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.linear_model import Ridge
import re
import numpy as np
from PIL import Image, ImageDraw
from textblob import TextBlob
import io

# Page Config
st.set_page_config(page_title="Fragrance Verbatim Lab Pro", layout="wide", page_icon="🧪")

# --- Multilingual Exclusion Dictionary ---
MULTILINGUAL_STOPWORDS = {
    "English": ["product", "smell", "feel", "really", "just", "like", "little", "think", "lot", "make", "also", "bit", "quite", "something", "seem", "evoke", "find", "remind"],
    "French": ["produit", "odeur", "sent", "vraiment", "comme", "plus", "bien", "fait", "tout", "après", "assez", "évoque", "trouve", "rappelle", "petit", "beaucoup", "être", "avoir"],
    "German": ["produkt", "riecht", "geruch", "wirklich", "ganz", "viel", "mehr", "oder", "etwa", "lässt", "erinnert", "finde", "bisschen", "scheint", "etwas", "gut", "immer"],
    "Spanish": ["producto", "huele", "olor", "muy", "como", "mas", "pero", "todo", "este", "sentir", "parece", "evoca", "encuentro", "recuerda", "poco", "mucho", "bien"],
    "Portuguese": ["producto", "cheiro", "sinto", "muito", "como", "mais", "mas", "tudo", "este", "parece", "evoca", "acho", "lembra", "pouco", "muito", "bem"],
    "Italian": ["prodotto", "odore", "sento", "molto", "come", "più", "ma", "tutto", "questo", "sembra", "evoca", "trovo", "ricorda", "poco", "molto", "bene"],
    "Indonesian": ["produk", "bau", "wangi", "sangat", "seperti", "lebih", "tapi", "semua", "ini", "merasa", "tampak", "mengingatkan", "sedikit", "banyak", "bagus"]
}

# --- NLP Engine ---
@st.cache_resource
def setup_nltk():
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('stopwords', quiet=True)
    return WordNetLemmatizer()

lemmatizer = setup_nltk()

def clean_text(text, custom_stops, lang_choice, gram_rules):
    if not text or pd.isna(text): return ""
    lang_map = {"English": "english", "French": "french", "German": "german", "Spanish": "spanish", "Portuguese": "portuguese", "Italian": "italian", "Indonesian": "indonesian"}
    try:
        base_stops = set(nltk.corpus.stopwords.words(lang_map.get(lang_choice, "english")))
    except:
        base_stops = set()

    custom_stops_set = set([str(x).strip().lower() for x in custom_stops])

    # FIX 1: Include negation_list and superlative_list words in gram_influencers
    # so they survive stopword filtering and can form grams properly
    gram_influencers = set(
        gram_rules['prefix_2g'] +
        gram_rules['suffix_2g'] +
        [w for phrase in gram_rules['prefix_3g'] for w in phrase.split()] +
        [w for phrase in gram_rules['spec_2g'] for w in phrase.split()] +
        [w for phrase in gram_rules['spec_3g'] for w in phrase.split()] +
        # ADDED: protect negation and superlative seed words from being dropped
        [w for phrase in gram_rules['negation_list'] for w in phrase.split()] +
        [w for phrase in gram_rules['superlative_list'] for w in phrase.split()]
    )

    fragrance_merges = {"freshness": "fresh", "freshly": "fresh", "fruity": "fruit", "smelling": "smell", "scented": "scent", "floral": "flower", "flowers": "flower", "cleanliness": "clean", "cleaning": "clean"}

    words = re.findall(r'\b[a-zà-ÿ]{2,}\b', str(text).lower())

    tokens = []
    for w in words:
        lemma = lemmatizer.lemmatize(w)
        lemma = fragrance_merges.get(lemma, lemma)

        if lemma in custom_stops_set:
            if lemma not in gram_influencers:
                continue
            tokens.append(lemma)
        elif lemma not in base_stops or lemma in gram_influencers:
            tokens.append(lemma)

    if not tokens: return ""

    processed_tokens = []
    i = 0
    while i < len(tokens):
        match_found = False
        if i < len(tokens) - 2:
            trigram_raw = f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}"
            prefix_2g = f"{tokens[i]} {tokens[i+1]}"
            if trigram_raw in gram_rules['spec_3g'] or prefix_2g in gram_rules['prefix_3g']:
                processed_tokens.append(f"{tokens[i]}_{tokens[i+1]}_{tokens[i+2]}")
                i += 3
                match_found = True
        if not match_found and i < len(tokens) - 1:
            bigram_raw = f"{tokens[i]} {tokens[i+1]}"
            if (bigram_raw in gram_rules['spec_2g'] or
                tokens[i] in gram_rules['prefix_2g'] or
                tokens[i+1] in gram_rules['suffix_2g']):
                processed_tokens.append(f"{tokens[i]}_{tokens[i+1]}")
                i += 2
                match_found = True

        if not match_found:
            if tokens[i] not in custom_stops_set:
                processed_tokens.append(tokens[i])
            i += 1

    return " ".join(processed_tokens)

def get_sentiment_words(text_series):
    words = " ".join(text_series).split()
    if not words: return [], []
    unique_words = list(set(words))
    scored = []
    for w in unique_words:
        display_text = w.replace("_", " ")
        score = TextBlob(display_text).sentiment.polarity
        scored.append((w, score))
    pos = sorted([x for x in scored if x[1] > 0.1], key=lambda x: x[1], reverse=True)[:10]
    neg = sorted([x for x in scored if x[1] < -0.1], key=lambda x: x[1])[:10]
    return pos, neg

def get_gram_categories(text_series, negation_prefixes, superlative_prefixes):
    """
    FIX 2: Corrected prefix matching logic.
    - Handles both single-word and multi-word prefixes (normalized to underscore form).
    - Also catches single-word tokens that are themselves in the lists (not just grams).
    - Avoids double-counting by checking negation first, then superlative.
    """
    words = " ".join(text_series).split()
    neg_captured = []
    sup_captured = []

    # Normalize prefixes to underscore form for consistent matching
    neg_p = [p.strip().lower().replace(" ", "_") for p in negation_prefixes]
    sup_p = [p.strip().lower().replace(" ", "_") for p in superlative_prefixes]

    for w in set(words):
        if "_" in w:
            # Gram token: check if it starts with any negation prefix
            if any(w.startswith(p + "_") or w == p for p in neg_p):
                neg_captured.append(w.replace("_", " "))
            # Only check superlative if not already a negation gram
            elif any(w.startswith(p + "_") or w == p for p in sup_p):
                sup_captured.append(w.replace("_", " "))
        else:
            # Single-word token: check direct membership in prefix lists
            if w in neg_p:
                neg_captured.append(w)
            elif w in sup_p:
                sup_captured.append(w)

    return sorted(list(set(neg_captured)))[:10], sorted(list(set(sup_captured)))[:10]

def generate_word_cloud(text_series, palette, shape):
    """
    FIX 3: Replace underscores with spaces before rendering so gram tokens
    like 'not_fresh' display as 'not fresh' in the word cloud.
    """
    # Convert underscored gram tokens to spaced phrases for display
    display_series = text_series.str.replace("_", " ", regex=False)
    combined_text = " ".join(display_series).strip()
    if not combined_text:
        fig, ax = plt.subplots(); ax.text(0.5, 0.5, "No text available", ha='center'); ax.axis("off")
        return fig
    mask = None
    if shape == "Round":
        img = Image.new("L", (800, 800), 255)
        draw = ImageDraw.Draw(img); draw.ellipse((20,20,780,780), fill=0); mask = np.array(img)
    wc = WordCloud(background_color="white", colormap=palette, mask=mask, width=800, height=500, collocations=False)
    wc.generate(combined_text)
    fig, ax = plt.subplots(); ax.imshow(wc, interpolation='bilinear'); ax.axis("off")
    return fig

def generate_word_tree(text_series, min_freq, palette):
    valid = [t for t in text_series if len(t.split()) > 1]
    if not valid: return None
    try:
        vec = CountVectorizer(min_df=min_freq)
        mtx = vec.fit_transform(valid); words = vec.get_feature_names_out()
        if len(words) < 2: return None
        adj = (mtx.T * mtx); adj.setdiag(0); G = nx.from_scipy_sparse_array(adj)
        G = nx.relabel_nodes(G, {i: w for i, w in enumerate(words)})
        T = nx.maximum_spanning_tree(G)
        fig, ax = plt.subplots(figsize=(8,6))
        pos = nx.spring_layout(T, k=1.5, seed=42); part = community_louvain.best_partition(T)
        nx.draw_networkx_nodes(T, pos, node_size=2000, node_color=list(part.values()), cmap=palette, alpha=0.8)
        nx.draw_networkx_labels(T, pos, font_size=8, font_weight='bold'); nx.draw_networkx_edges(T, pos, alpha=0.2)
        plt.axis('off'); return fig
    except: return None

def run_fca(df, p_col, fmin, use_tfidf):
    grouped = df.groupby(p_col)['cleaned'].apply(lambda x: " ".join(x))
    if len(grouped) < 3: return None, "Need 3+ products for Factorial Mapping."
    VecClass = TfidfVectorizer if use_tfidf else CountVectorizer
    vec = VecClass(min_df=min(fmin, len(grouped)))
    X = vec.fit_transform(grouped).toarray()
    words, products = vec.get_feature_names_out(), grouped.index.tolist()
    X_centered = X - np.mean(X, axis=0)
    svd = TruncatedSVD(n_components=2, random_state=42)
    row_coords = svd.fit_transform(X_centered)
    col_coords = svd.components_.T * (np.std(row_coords) / (np.std(svd.components_.T) + 1e-9))
    return (row_coords, col_coords, products, words, svd.explained_variance_ratio_), None

# --- UI Setup ---
with st.sidebar:
    st.header("⚙️ Settings")
    uploaded_file = st.file_uploader("Upload Excel", type=["xlsx"])

    if uploaded_file:
        try:
            xl = pd.ExcelFile(uploaded_file)
            sheet = st.selectbox("Select Sheet:", xl.sheet_names)
            df_raw = pd.read_excel(uploaded_file, sheet_name=sheet)

            st.subheader("🎯 Sub-Target Filter")
            filter_col = st.selectbox("Filter Column:", ["No Filter"] + list(df_raw.columns))
            target_indices = df_raw.index
            filter_label = "Total Sample"
            if filter_col != "No Filter":
                options = sorted(df_raw[filter_col].dropna().unique())
                selected_codes = st.multiselect("Select Codes:", options)
                if selected_codes:
                    target_indices = df_raw[df_raw[filter_col].isin(selected_codes)].index
                    filter_label = f"{filter_col}: {', '.join(map(str, selected_codes))}"
        except ImportError:
            st.error("❌ The 'openpyxl' library is missing. Please add it to your requirements.txt file.")
            st.stop()

        st.divider()
        dataset_lang = st.selectbox("Language:", list(MULTILINGUAL_STOPWORDS.keys()))
        if 'custom_stop_list' not in st.session_state:
            st.session_state.custom_stop_list = MULTILINGUAL_STOPWORDS[dataset_lang]

        fmin_global = st.slider("Min Word Frequency", 1, 50, 5)
        use_tfidf = st.toggle("Use TF-IDF Weighting", value=True)
        shape_opt = st.radio("Cloud Shape", ["Rectangle", "Round"])
        palette_opt = st.selectbox("Palette", ["copper", "GnBu", "RdPu", "viridis"])

if 'gram_rules' not in st.session_state:
    st.session_state.gram_rules = {
        'prefix_2g': ["not", "too", "very", "real", "really", "enough", "because", "if", "less", "more", "little", "lot", "all", "so", "just", "quite", "many"],
        'suffix_2g': ["not", "too", "very", "real", "really", "enough", "because", "if", "less", "more", "little", "lot", "all", "so", "quite"],
        'prefix_3g': ["not too", "not very", "not real", "not enough"],
        'spec_2g': ["lily valley", "funeral flower", "white flower", "old fashion", "old people", "old lady", "house cleaner", "not fresh", "not clean"],
        'spec_3g': ["not smell good", "smell very good", "not smell bad", "smell very bad"],
        # Lists used for categorising gram descriptors in the summary panels
        'negation_list': ["not", "not too", "less", "little", "not very", "not at all"],
        'superlative_list': ["really", "very", "enough", "quite", "many", "just", "more", "real", "so", "too", "too much"]
    }

tab1, tab2, tab3, tab4, tab6, tab5 = st.tabs(["📊 Single Product", "⚔️ Comparison", "🌐 Factorial Map", "🔍 Topic Lab", "🎯 Impact Lab", "🚫 Exclusions & Grams"])

if uploaded_file and 'df_raw' in locals():
    p_col = st.sidebar.selectbox("Product ID Column", df_raw.columns)
    v_col = st.sidebar.selectbox("Verbatim Column", df_raw.columns)
    s_col = st.sidebar.selectbox("Preference Score (Optional)", ["None"] + list(df_raw.columns))

    if st.sidebar.button("🚀 Run Analysis on Sub-Target"):
        df_filtered = df_raw.loc[target_indices].dropna(subset=[v_col])
        df_filtered['cleaned'] = df_filtered[v_col].apply(lambda x: clean_text(x, st.session_state.custom_stop_list, dataset_lang, st.session_state.gram_rules))
        st.session_state['processed_df'] = df_filtered
        st.session_state['filter_info'] = filter_label
        st.session_state['pref_col'] = s_col

    if 'processed_df' in st.session_state:
        df = st.session_state['processed_df']
        p_list = sorted(df[p_col].dropna().astype(str).unique())
        st.caption(f"📍 **Currently Analyzing:** {st.session_state.get('filter_info', 'Total Sample')} (N={len(df)})")

        with tab1:
            target_p = st.selectbox("Fragrance Focus", p_list)
            product_data = df[df[p_col].astype(str) == target_p]
            p_sub_cleaned = product_data['cleaned']

            if not p_sub_cleaned.empty:
                full_text = " ".join(p_sub_cleaned)
                cv = CountVectorizer()
                cv_mtx = cv.fit_transform([full_text])
                counts = dict(zip(cv.get_feature_names_out(), cv_mtx.toarray()[0]))
                tv = TfidfVectorizer()
                tv_mtx = tv.fit_transform([full_text])
                tfidf = dict(zip(tv.get_feature_names_out(), tv_mtx.toarray()[0]))

                export_df = pd.DataFrame({"Word": counts.keys(),"Unweighted Frequency": counts.values(),"Weighted (TF-IDF) Frequency": [tfidf[w] for w in counts.keys()]}).sort_values(by="Unweighted Frequency", ascending=False)
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    export_df.to_excel(writer, index=False, sheet_name='Word Frequencies')
                st.download_button(label="📥 Download Word Cloud Stats (Excel)", data=output.getvalue(), file_name=f"{target_p}_word_cloud_stats.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            sent_val = product_data[v_col].apply(lambda x: TextBlob(str(x)).sentiment.polarity).mean()
            st.metric(f"Target Mood: {target_p}", f"{'Positive' if sent_val > 0 else 'Negative'}", f"{round(sent_val*100, 1)}%")
            st.progress((sent_val + 1) / 2)
            c1, c2 = st.columns(2)
            with c1:
                # FIX 3 applied here: underscores replaced inside generate_word_cloud
                st.pyplot(generate_word_cloud(p_sub_cleaned, palette_opt, shape_opt))
            with c2:
                tree_fig = generate_word_tree(p_sub_cleaned, fmin_global, palette_opt)
                if tree_fig: st.pyplot(tree_fig)
                else: st.warning("Not enough patterns.")

            # Row 1: Sentiments
            pos_words, neg_words = get_sentiment_words(p_sub_cleaned)
            l, r = st.columns(2)
            with l:
                st.success("✨ **Positive Descriptors**")
                for w, s in pos_words: st.write(f"- {w.replace('_', ' ')}")
            with r:
                st.error("⚠️ **Negative Descriptors**")
                for w, s in neg_words: st.write(f"- {w.replace('_', ' ')}")

            # Row 2: Gram Categories (Negation & Superlative)
            # FIX 2 applied: get_gram_categories now correctly matches prefixes
            neg_grams, sup_grams = get_gram_categories(
                p_sub_cleaned,
                st.session_state.gram_rules['negation_list'],
                st.session_state.gram_rules['superlative_list']
            )
            l2, r2 = st.columns(2)
            with l2:
                st.warning("🚫 **Negation Gram**")
                if neg_grams:
                    for g in neg_grams: st.write(f"- {g}")
                else: st.write("No negations found.")
            with r2:
                st.info("💎 **Superlative**")
                if sup_grams:
                    for g in sup_grams: st.write(f"- {g}")
                else: st.write("No superlatives found.")

        # --- Remaining tabs ---
        with tab2:
            st.subheader("⚔️ Scent Comparison")
            comp_cols = st.columns(2)
            p_a = comp_cols[0].selectbox("Fragrance A", p_list, index=0)
            p_b = comp_cols[1].selectbox("Fragrance B", p_list, index=min(1, len(p_list)-1))
            d_a = df[df[p_col].astype(str) == p_a]['cleaned']
            d_b = df[df[p_col].astype(str) == p_b]['cleaned']
            if not d_a.empty and not d_b.empty:
                sim = float(cosine_similarity(TfidfVectorizer().fit_transform([" ".join(d_a), " ".join(d_b)]))[0][1])
                st.metric("Olfactive Similarity", f"{round(sim*100, 1)}%")
                st.progress(sim)
                comp_cols[0].pyplot(generate_word_cloud(d_a, palette_opt, shape_opt))
                comp_cols[1].pyplot(generate_word_cloud(d_b, palette_opt, shape_opt))

        with tab3:
            st.subheader("🌐 Factorial Mapping")
            res, err = run_fca(df, p_col, fmin_global, use_tfidf)
            if not err:
                r_c, c_c, prods, wrds, _ = res
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(r_c[:,0], r_c[:,1], c='blue', s=100)
                for i, txt in enumerate(prods): ax.text(r_c[i,0], r_c[i,1], txt, fontsize=12)
                ax.scatter(c_c[:,0], c_c[:,1], c='red', marker='x', alpha=0.2)
                for i, txt in enumerate(wrds):
                    if np.linalg.norm(c_c[i]) > np.percentile([np.linalg.norm(c) for c in c_c], 80):
                        ax.text(c_c[i,0], c_c[i,1], txt, color='darkred', fontsize=8)
                st.pyplot(fig)
            else: st.error(err)

        with tab4:
            st.subheader("🔍 Topic Lab")
            num_t = st.slider("Themes", 2, 8, 3)
            if st.button("Generate Topics"):
                vec = TfidfVectorizer(max_features=500)
                mtx = vec.fit_transform(df['cleaned'])
                nmf = NMF(n_components=num_t, random_state=42, init='nndsvd').fit(mtx)
                doc_topic = nmf.transform(mtx)
                fn = vec.get_feature_names_out()
                cols = st.columns(num_t)
                for i, topic in enumerate(nmf.components_):
                    with cols[i % num_t]:
                        top_words = [fn[j] for j in topic.argsort()[-7:]]
                        st.info(f"**Theme {i+1}**\n\n" + ", ".join(top_words))
                        closest_idx = doc_topic[:, i].argmax()
                        furthest_idx = doc_topic[:, i].argmin()
                        st.success(f"✅ **Closest:** {df.iloc[closest_idx][p_col]}")
                        st.error(f"❌ **Furthest:** {df.iloc[furthest_idx][p_col]}")

        with tab6:
            st.subheader("🎯 Preference Driver Analysis")
            pref_col = st.session_state.get('pref_col', "None")
            if pref_col == "None":
                st.warning("Please select a Preference Score column in the sidebar to unlock this tab.")
            else:
                try:
                    df_imp = df.dropna(subset=[pref_col, 'cleaned'])
                    df_imp = df_imp[df_imp['cleaned'] != ""]
                    vec_imp = CountVectorizer(min_df=3, binary=True)
                    X_imp = vec_imp.fit_transform(df_imp['cleaned'])
                    y_imp = df_imp[pref_col]
                    model = Ridge(alpha=1.0).fit(X_imp, y_imp)
                    impact_df = pd.DataFrame({'Word': vec_imp.get_feature_names_out(), 'Impact': model.coef_}).sort_values(by='Impact', ascending=False)

                    c1, c2 = st.columns(2)
                    with c1:
                        st.success("**Positive Drivers**")
                        st.dataframe(impact_df.head(10))
                    with c2:
                        st.error("**Negative Drivers**")
                        st.dataframe(impact_df.tail(10))

                    fig, ax = plt.subplots(figsize=(10, 6))
                    top_bot = pd.concat([impact_df.head(10), impact_df.tail(10)])
                    ax.barh(top_bot['Word'], top_bot['Impact'], color=['green' if x > 0 else 'red' for x in top_bot['Impact']])
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error calculating drivers: {e}")

with tab5:
    st.subheader("🚫 Exclusions & Gram Lab")
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("### 🛑 Word Exclusions")
        stops = st.session_state.get('custom_stop_list', [])
        txt_stops = st.text_area("Stopwords (comma separated)", value=", ".join(stops), height=150)

        st.markdown("### 📊 List Category Prefixes")
        gn_list = st.text_input("Grams Negation for Lists", ", ".join(st.session_state.gram_rules['negation_list']))
        gs_list = st.text_input("Grams Superlative for Lists", ", ".join(st.session_state.gram_rules['superlative_list']))

    with col_right:
        st.markdown("### 🔗 Gram Dictionary")
        g = st.session_state.gram_rules
        p2 = st.text_input("Word prefix of authorized 2-gram", ", ".join(g['prefix_2g']))
        s2 = st.text_input("Word suffix of authorized 2-gram", ", ".join(g['suffix_2g']))
        p3 = st.text_input("2-gram prefix of authorized 3-gram", ", ".join(g['prefix_3g']))
        a2 = st.text_input("Special authorization of 2-gram", ", ".join(g['spec_2g']))
        a3 = st.text_input("Special authorization of 3-gram", ", ".join(g['spec_3g']))

    if st.button("💾 Apply Rules & Re-Process"):
        st.session_state.custom_stop_list = [x.strip().lower() for x in txt_stops.split(",") if x.strip()]
        st.session_state.gram_rules = {
            'prefix_2g': [x.strip().lower() for x in p2.split(",") if x.strip()],
            'suffix_2g': [x.strip().lower() for x in s2.split(",") if x.strip()],
            'prefix_3g': [x.strip().lower() for x in p3.split(",") if x.strip()],
            'spec_2g': [x.strip().lower() for x in a2.split(",") if x.strip()],
            'spec_3g': [x.strip().lower() for x in a3.split(",") if x.strip()],
            'negation_list': [x.strip().lower() for x in gn_list.split(",") if x.strip()],
            'superlative_list': [x.strip().lower() for x in gs_list.split(",") if x.strip()]
        }
        st.rerun()
