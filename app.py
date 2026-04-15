import streamlit as st
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
import networkx as nx
from community import community_louvain
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial import ConvexHull
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.linear_model import Ridge
import re
import numpy as np
from PIL import Image, ImageDraw
from textblob import TextBlob
import io
from collections import Counter

# Page Config
st.set_page_config(page_title="Fragrance Verbatim Lab Pro", layout="wide", page_icon="🧪")

# --- Comprehensive Multilingual Packs ---
LANGUAGE_PACKS = {
    "English": {
        "stops": ["product", "smell", "feel", "really", "just", "like", "little", "think", "lot", "make", "also", "bit", "quite", "something", "seem", "evoke", "find", "remind"],
        "negation": ["not", "not too", "less", "little", "not very", "not at all", "no"],
        "superlative": ["really", "very", "enough", "quite", "many", "just", "more", "real", "so", "too"],
        "rules": {
            "prefix_2g": ["not", "too", "very", "real", "really", "enough", "less", "more", "little", "lot", "so", "just", "quite", "many", "no"],
            "suffix_2g": ["enough", "away"],
            "prefix_3g": ["not too", "not very", "not real", "not enough"],
            "spec_2g": ["lily valley", "funeral flower", "white flower", "old fashion", "old people", "old lady", "house cleaner", "not fresh", "not clean"],
            "spec_3g": ["not smell good", "smell very good", "not smell bad", "smell very bad"]
        }
    },
    "French": {
        "stops": ["produit", "odeur", "sentir", "vraiment", "juste", "comme", "petit", "penser", "beaucoup", "faire", "aussi", "peu", "assez", "quelque chose", "sembler", "évoquer", "trouver", "rappeler"],
        "negation": ["pas", "pas trop", "moins", "peu", "pas très", "pas du tout", "non plus"],
        "superlative": ["vraiment", "très", "assez", "plutôt", "beaucoup", "juste", "plus", "réel", "tellement", "trop"],
        "rules": {
            "prefix_2g": ["pas", "trop", "très", "vrai", "vraiment", "assez", "moins", "plus", "peu", "beaucoup", "tellement", "juste", "plutôt", "plusieurs", "non"],
            "suffix_2g": ["assez", "partout"],
            "prefix_3g": ["pas trop", "pas très", "pas vraiment", "pas assez"],
            "spec_2g": ["muguet", "fleur de cimetière", "fleurs blanches", "démodé", "personnes âgées", "vieille dame", "produit ménager", "manque de fraîcheur", "pas clair"],
            "spec_3g": ["ne sent pas bon", "sent super bon", "ne sent pas mauvais", "sent très mauvais"]
        }
    },
    "Spanish": {
        "stops": ["producto", "olor", "sentir", "realmente", "solo", "como", "poco", "pensar", "mucho", "hacer", "también", "un poco", "bastante", "algo", "parecer", "evocar", "encontrar", "recordar"],
        "negation": ["no", "no demasiado", "menos", "poco", "no muy", "para nada", "tampoco"],
        "superlative": ["realmente", "muy", "suficiente", "bastante", "muchos", "solo", "más", "real", "tan", "demasiado"],
        "rules": {
            "prefix_2g": ["no", "demasiado", "muy", "real", "realmente", "suficiente", "menos", "más", "poco", "mucho", "tan", "solo", "bastante", "varios", "ninguno"],
            "suffix_2g": ["suficiente", "lejos"],
            "prefix_3g": ["no demasiado", "no muy", "no realmente", "no suficiente"],
            "spec_2g": ["lirio de los valles", "flores de cementerio", "flores blancas", "pasado de moda", "gente mayor", "anciana", "limpiador de hogar", "poco fresco", "nada claro"],
            "spec_3g": ["no huele bien", "huele de maravilla", "no huele mal", "huele fatal"]
        }
    }
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
    gram_influencers = set(
        gram_rules['prefix_2g'] + gram_rules['suffix_2g'] +
        [w for phrase in gram_rules['prefix_3g'] for w in phrase.split()] +
        [w for phrase in gram_rules['spec_2g'] for w in phrase.split()] +
        [w for phrase in gram_rules['spec_3g'] for w in phrase.split()] +
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
            if lemma not in gram_influencers: continue
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
            prefix_2g_part = f"{tokens[i]} {tokens[i+1]}"
            if trigram_raw in gram_rules['spec_3g'] or prefix_2g_part in gram_rules['prefix_3g']:
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

def generate_word_cloud(text_series, palette, shape):
    combined_text = " ".join(text_series).strip()
    if not combined_text:
        fig, ax = plt.subplots(); ax.text(0.5, 0.5, "No text available", ha='center'); ax.axis("off")
        return fig
    mask = None
    if shape == "Round":
        img = Image.new("L", (800, 800), 255)
        draw = ImageDraw.Draw(img); draw.ellipse((20,20,780,780), fill=0); mask = np.array(img)
    wc = WordCloud(background_color="white", colormap=palette, mask=mask, width=800, height=500, collocations=False, regexp=r"\S+").generate(combined_text)
    fig, ax = plt.subplots(); ax.imshow(wc, interpolation='bilinear'); ax.axis("off")
    return fig

def generate_word_tree_advanced(text_series, min_freq, palette):
    """
    High-quality Word Tree matching Target style:
    - Well-separated clusters with distinct pastel colors
    - Clean non-overlapping convex hull regions
    - Font size proportional to frequency
    - No bounding boxes on text
    """
    valid = [t for t in text_series if len(str(t).split()) > 0]
    if not valid:
        return None
    try:
        vec = CountVectorizer(min_df=min_freq, token_pattern=r"(?u)\b\S+\b")
        mtx = vec.fit_transform(valid)
        words = vec.get_feature_names_out()
        word_counts = np.asarray(mtx.sum(axis=0)).flatten()
        count_dict = dict(zip(words, word_counts))
        if len(words) < 2:
            return None

        adj = (mtx.T * mtx)
        adj.setdiag(0)
        G = nx.from_scipy_sparse_array(adj)
        G = nx.relabel_nodes(G, {i: w for i, w in enumerate(words)})

        # Remove very weak edges to get cleaner cluster separation
        weak_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('weight', 1) < 2]
        G.remove_edges_from(weak_edges)

        # Remove isolated nodes after pruning
        G.remove_nodes_from(list(nx.isolates(G)))

        if len(G.nodes) < 2:
            return None

        partition = community_louvain.best_partition(G)

        # High k value for strong geographic separation between clusters
        pos = nx.spring_layout(G, k=4.0, seed=42, iterations=300)

        fig, ax = plt.subplots(figsize=(14, 10), facecolor='white')
        ax.set_facecolor('white')

        # Distinct pastel colors matching the Target aesthetic
        PASTEL_COLORS = [
            "#A8D8B9",  # soft green
            "#F4B8C1",  # soft pink/rose
            "#B5D0E8",  # soft blue
            "#D4E8A8",  # soft yellow-green
            "#C8B8E8",  # soft purple/lavender
            "#F4D8A8",  # soft peach/amber
            "#A8D8D8",  # soft teal
            "#E8C8B8",  # soft coral
        ]

        unique_comms = sorted(list(set(partition.values())))

        # --- Draw hull regions FIRST (behind everything) ---
        for i, comm in enumerate(unique_comms):
            nodes_in_comm = [n for n in G.nodes() if partition[n] == comm]
            if not nodes_in_comm:
                continue

            color = PASTEL_COLORS[i % len(PASTEL_COLORS)]

            # Estimate text extents for padding hull around actual rendered text
            pts = []
            max_c = max(word_counts) if len(word_counts) > 0 else 1
            for n in nodes_in_comm:
                x, y = pos[n]
                fsize = 9 + (count_dict.get(n, 1) / max_c) * 22
                char_w = fsize * 0.55
                text_w = len(n.replace("_", "\n").split("\n")[0]) * char_w
                text_h = fsize * 1.2

                # Add corner points of approximate text bounding box
                pts.append([x - text_w / 2, y - text_h / 2])
                pts.append([x + text_w / 2, y - text_h / 2])
                pts.append([x - text_w / 2, y + text_h / 2])
                pts.append([x + text_w / 2, y + text_h / 2])

            pts = np.array(pts)

            if len(nodes_in_comm) >= 2 and len(pts) >= 4:
                # Expand outward from centroid for hull padding
                cent = np.mean(pts, axis=0)
                expanded_pts = pts + 0.12 * (pts - cent)

                try:
                    hull = ConvexHull(expanded_pts)
                    hull_verts = expanded_pts[hull.vertices]

                    polygon = patches.Polygon(
                        hull_verts,
                        closed=True,
                        facecolor=color,
                        edgecolor=color,
                        alpha=0.35,
                        linewidth=1.5,
                        zorder=0,
                        joinstyle='round',
                        capstyle='round'
                    )
                    ax.add_patch(polygon)
                except Exception:
                    center = np.mean(pts, axis=0)
                    radius = np.max(np.linalg.norm(pts - center, axis=1)) + 0.05
                    circle = plt.Circle(center, radius, color=color, alpha=0.3, zorder=0)
                    ax.add_artist(circle)
            elif nodes_in_comm:
                center = np.mean(pts, axis=0)
                circle = plt.Circle(center, 0.12, color=color, alpha=0.3, zorder=0)
                ax.add_artist(circle)

        # --- Draw edges (thin, light grey) ---
        weights = [G[u][v].get('weight', 1) for u, v in G.edges()]
        if weights:
            max_w = max(weights) if max(weights) > 0 else 1
            norm_widths = [(w / max_w) * 3 + 0.3 for w in weights]
            nx.draw_networkx_edges(
                G, pos,
                width=norm_widths,
                alpha=0.25,
                edge_color='#aaaaaa',
                ax=ax,
                style='solid'
            )

        # --- Draw word labels (clean, no boxes) ---
        max_c = max(word_counts) if len(word_counts) > 0 else 1
        for node, (x, y) in pos.items():
            freq_ratio = count_dict.get(node, 1) / max_c
            fsize = 9 + freq_ratio * 22  # range: 9px (rare) to 31px (most common)
            weight = 'bold' if freq_ratio > 0.3 else 'normal'
            display_text = node.replace("_", " ")

            ax.text(
                x, y,
                display_text,
                fontsize=fsize,
                fontweight=weight,
                ha='center',
                va='center',
                color='#111111',
                zorder=3,
            )

        ax.set_xlim(ax.get_xlim()[0] - 0.3, ax.get_xlim()[1] + 0.3)
        ax.set_ylim(ax.get_ylim()[0] - 0.3, ax.get_ylim()[1] + 0.3)
        plt.axis('off')
        plt.tight_layout(pad=0.5)
        return fig

    except Exception as e:
        print(f"Word tree error: {e}")
        return None

def run_fca(df, p_col, fmin, use_tfidf):
    grouped = df.groupby(p_col)['cleaned'].apply(lambda x: " ".join(x))
    if len(grouped) < 3: return None, "Need 3+ products for Factorial Mapping."
    VecClass = TfidfVectorizer if use_tfidf else CountVectorizer
    vec = VecClass(min_df=min(fmin, len(grouped)), token_pattern=r"(?u)\b\S+\b")
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
    dataset_lang = st.selectbox("Language:", list(LANGUAGE_PACKS.keys()))

    pack = LANGUAGE_PACKS[dataset_lang]
    if 'current_lang' not in st.session_state or st.session_state.current_lang != dataset_lang:
        st.session_state.current_lang = dataset_lang
        st.session_state.custom_stop_list = pack['stops']
        st.session_state.gram_rules = {
            'prefix_2g': pack['rules']['prefix_2g'],
            'suffix_2g': pack['rules']['suffix_2g'],
            'prefix_3g': pack['rules']['prefix_3g'],
            'spec_2g': pack['rules']['spec_2g'],
            'spec_3g': pack['rules']['spec_3g'],
            'negation_list': pack['negation'],
            'superlative_list': pack['superlative']
        }

    if uploaded_file:
        try:
            xl = pd.ExcelFile(uploaded_file)
            sheet = st.selectbox("Select Sheet:", xl.sheet_names)
            df_raw = pd.read_excel(uploaded_file, sheet_name=sheet)
            filter_col = st.selectbox("Filter Column:", ["No Filter"] + list(df_raw.columns))
            target_indices = df_raw.index
            filter_label = "Total Sample"
            if filter_col != "No Filter":
                options = sorted(df_raw[filter_col].dropna().unique())
                selected_codes = st.multiselect("Select Codes:", options)
                if selected_codes:
                    target_indices = df_raw[df_raw[filter_col].isin(selected_codes)].index
                    filter_label = f"{filter_col}: {', '.join(map(str, selected_codes))}"
        except Exception as e: st.error(f"Error: {e}"); st.stop()

    fmin_global = st.slider("Min Word Frequency", 1, 50, 5)
    use_tfidf = st.toggle("Use TF-IDF Weighting", value=True)
    shape_opt = st.radio("Cloud Shape", ["Rectangle", "Round"])
    palette_opt = st.selectbox("Palette", ["copper", "GnBu", "RdPu", "viridis", "Spectral"])

tab1, tab2, tab3, tab4, tab6, tab5 = st.tabs(["📊 Single Product", "⚔️ Comparison", "🌐 Factorial Map", "🔍 Topic Lab", "🎯 Impact Lab", "🚫 Exclusions & Grams"])

if uploaded_file and 'df_raw' in locals():
    p_col = st.sidebar.selectbox("Product ID Column", df_raw.columns)
    v_col = st.sidebar.selectbox("Verbatim Column", df_raw.columns)
    s_col = st.sidebar.selectbox("Preference Score (Optional)", ["None"] + list(df_raw.columns))

    if st.sidebar.button("🚀 Run Analysis"):
        df_filtered = df_raw.loc[target_indices].dropna(subset=[v_col])
        df_filtered['cleaned'] = df_filtered[v_col].apply(lambda x: clean_text(x, st.session_state.custom_stop_list, dataset_lang, st.session_state.gram_rules))
        st.session_state['processed_df'] = df_filtered
        st.session_state['filter_info'] = filter_label
        st.session_state['pref_col'] = s_col

    if 'processed_df' in st.session_state:
        df = st.session_state['processed_df']
        p_list = sorted(df[p_col].dropna().astype(str).unique())

        with tab1:
            target_p = st.selectbox("Fragrance Focus", p_list)
            product_data = df[df[p_col].astype(str) == target_p]
            p_sub_cleaned = product_data['cleaned']

            sent_val = product_data[v_col].apply(lambda x: TextBlob(str(x)).sentiment.polarity).mean()
            st.metric(f"Mood: {target_p}", f"{'Positive' if sent_val > 0 else 'Negative'}", f"{round(sent_val*100, 1)}%")

            st.write("### 🌳 Olfactive Word Tree")
            tree_fig = generate_word_tree_advanced(p_sub_cleaned, fmin_global, palette_opt)
            if tree_fig: st.pyplot(tree_fig)
            else: st.warning("Not enough data for tree.")

            st.divider()
            st.write("### ☁️ Classic Wordcloud")
            st.pyplot(generate_word_cloud(p_sub_cleaned, palette_opt, shape_opt))

        with tab2:
            st.subheader("⚔️ Scent Comparison")
            comp_cols = st.columns(2)
            p_a = comp_cols[0].selectbox("Fragrance A", p_list, index=0)
            p_b = comp_cols[1].selectbox("Fragrance B", p_list, index=min(1, len(p_list)-1))
            d_a, d_b = df[df[p_col].astype(str) == p_a]['cleaned'], df[df[p_col].astype(str) == p_b]['cleaned']
            if not d_a.empty and not d_b.empty:
                sim = float(cosine_similarity(TfidfVectorizer(token_pattern=r"(?u)\b\S+\b").fit_transform([" ".join(d_a), " ".join(d_b)]))[0][1])
                st.metric("Olfactive Similarity", f"{round(sim*100, 1)}%")
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
                        ax.text(c_c[i,0], c_c[i,1], txt.replace("_", " "), color='darkred', fontsize=8)
                st.pyplot(fig)
            else: st.error(err)

        with tab4:
            st.subheader("🔍 Topic Lab")
            num_t = st.slider("Themes", 2, 8, 3)
            if st.button("Generate Topics"):
                vec = TfidfVectorizer(max_features=500, token_pattern=r"(?u)\b\S+\b")
                mtx = vec.fit_transform(df['cleaned'])
                nmf = NMF(n_components=num_t, random_state=42, init='nndsvd').fit(mtx)
                doc_topic, fn = nmf.transform(mtx), vec.get_feature_names_out()
                cols = st.columns(num_t)
                for i, topic in enumerate(nmf.components_):
                    with cols[i % num_t]:
                        top_words = [fn[j].replace("_", " ") for j in topic.argsort()[-7:]]
                        st.info(f"**Theme {i+1}**\n\n" + ", ".join(top_words))

        with tab6:
            st.subheader("🎯 Preference Driver Analysis")
            pref_col = st.session_state.get('pref_col', "None")
            if pref_col != "None":
                try:
                    df_imp = df.dropna(subset=[pref_col, 'cleaned']).loc[lambda x: x['cleaned'] != ""]
                    vec_imp = CountVectorizer(min_df=3, binary=True, token_pattern=r"(?u)\b\S+\b")
                    X_imp, y_imp = vec_imp.fit_transform(df_imp['cleaned']), df_imp[pref_col]
                    model = Ridge(alpha=1.0).fit(X_imp, y_imp)
                    impact_df = pd.DataFrame({'Word': [w.replace("_", " ") for w in vec_imp.get_feature_names_out()], 'Impact': model.coef_}).sort_values(by='Impact', ascending=False)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.write("📈 Positive Drivers")
                        top10 = impact_df.head(10)
                        fig_pos, ax_pos = plt.subplots(figsize=(5, 4))
                        ax_pos.barh(top10['Word'], top10['Impact'], color='steelblue')
                        ax_pos.invert_yaxis()
                        plt.tight_layout()
                        st.pyplot(fig_pos)
                    with c2:
                        st.write("📉 Negative Drivers")
                        bot10 = impact_df.tail(10)
                        fig_neg, ax_neg = plt.subplots(figsize=(5, 4))
                        ax_neg.barh(bot10['Word'], bot10['Impact'], color='salmon')
                        ax_neg.invert_yaxis()
                        plt.tight_layout()
                        st.pyplot(fig_neg)
                except Exception as e: st.error(f"Error: {e}")
            else:
                st.info("Select a Preference Score column in the sidebar to enable this tab.")

        with tab5:
            st.subheader("🚫 Exclusions & Gram Lab")
            col_left, col_right = st.columns(2)
            stops = st.session_state.get('custom_stop_list', [])
            txt_stops = col_left.text_area("Stopwords", value=", ".join(stops), height=150)
            g = st.session_state.gram_rules
            p2 = col_right.text_input("Prefix 2-gram", ", ".join(g['prefix_2g']))
            a2 = col_right.text_input("Special 2-gram", ", ".join(g['spec_2g']))
            if st.button("💾 Apply & Re-Process"):
                st.session_state.custom_stop_list = [x.strip().lower() for x in txt_stops.split(",") if x.strip()]
                st.rerun()
