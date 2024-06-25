import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
from collections import Counter

# Set page config (this must be the first Streamlit command)
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")


# Load data
@st.cache_data
def load_data():
    movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
    movies = pd.DataFrame(movies_dict)
    similarity_metrics = pickle.load(open('similarity.pkl', 'rb'))
    return movies, similarity_metrics


movies, similarity_metrics = load_data()


def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distance = similarity_metrics[movie_index]
    movies_list = sorted(list(enumerate(distance)), reverse=True, key=lambda x: x[1])[1:7]

    recommended_movies = []
    for i in movies_list:
        movie_data = movies.iloc[i[0]]
        recommended_movies.append(movie_data)
    return recommended_movies


# Custom CSS
st.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
        font-weight: bold;
    }
    .medium-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .stSelectbox > div > div {
        background-color: #;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/clapperboard.png", width=100)
    st.markdown('<p class="big-font">Movie Recommender</p>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<p class="medium-font">About</p>', unsafe_allow_html=True)
    st.info(
        """
        This is a movie recommendation system built using a pre-trained model.
        Select a movie from the dropdown menu to get a list of similar movies.
        """
    )
    st.markdown("---")
    st.markdown("Made with ❤️ by TheTechieBoy")

# Main content
st.markdown('<p class="big-font">Discover Your Next Favorite Movie</p>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    movie_name_selection = st.selectbox(
        'Select your favorite movie',
        movies['title'].values
    )

if st.button('Get Recommendations', key='recommend_button'):
    with st.spinner('Finding great movies for you...'):
        recommendations = recommend(movie_name_selection)
        recommended_titles = [rec['title'] for rec in recommendations]

    st.success('Here are your personalized recommendations!')

    # Display recommendations
    st.markdown('<p class="medium-font">Top Picks for You</p>', unsafe_allow_html=True)
    for title in recommended_titles:
        st.write(f"🎬 {title}")

    # DataFrame for visualizations
    df_recommendations = pd.DataFrame(recommendations)

    # Tabs for visualizations
    tab1, tab2 = st.tabs(["📊 Tag Analysis", "🔍 Word Cloud"])

    with tab1:
        st.markdown('<p class="medium-font">Most Common Tags</p>', unsafe_allow_html=True)
        tags_text = ' '.join(df_recommendations['tags'].tolist())
        tags_counter = Counter(tags_text.split())
        common_tags = tags_counter.most_common(10)
        tags_df = pd.DataFrame(common_tags, columns=['Tag', 'Count'])

        fig = px.bar(tags_df, x='Tag', y='Count', title='Top 10 Tags')
        fig.update_layout(xaxis_title="Tag", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown('<p class="medium-font">Word Cloud of Movie Tags</p>', unsafe_allow_html=True)
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(tags_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)

else:
    st.info('Select a movie and click "Get Recommendations" to start exploring!')