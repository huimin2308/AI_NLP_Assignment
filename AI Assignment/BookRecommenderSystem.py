import streamlit as st
import streamlit.components.v1 as components
from streamlit_extras.stylable_container import stylable_container
import pandas as pd
import os
import string
from scipy.sparse import csr_matrix
from collections import defaultdict

# ---- Read relevant file ----
book_df = pd.read_csv("display_book_data.csv")
cleaned_book_df = pd.read_csv("cleaned_books_data.csv")
cleaned_review_df = pd.read_csv("cleaned_review_data.csv")

# ---- Mapping Usage ---- 
indices = pd.Series(cleaned_book_df.index, index=cleaned_book_df['title']).drop_duplicates(keep='first')
book_id_to_index = pd.Series(cleaned_book_df.index, index=cleaned_book_df['book_id']).drop_duplicates(keep='first')
index_to_book_id = pd.Series(cleaned_book_df['book_id'].values, index=cleaned_book_df.index)

# ---- Load all the relevant model ----
import pickle
bert_model = pickle.load(open('bert_model.pkl', 'rb'))
bert_embeddings = pickle.load(open('bert_embeddings.pkl', 'rb'))
bert_nn_model = pickle.load(open('bert_nn_model.pkl', 'rb'))
user_knn = pickle.load(open('user_knn_model.pkl', 'rb'))
user_item_matrix = pickle.load(open('user_item_matrix.pkl', 'rb'))
user_to_idx = pickle.load(open('user_to_idx.pkl', 'rb'))
book_to_idx = pickle.load(open('book_to_idx.pkl', 'rb'))
item_knn = pickle.load(open('item_knn_model.pkl', 'rb'))
item_item_df = pickle.load(open('item_item_matrix.pkl', 'rb'))
item_similarities = pickle.load(open('item_similarities.pkl', 'rb'))
item_indices = pickle.load(open('item_indices.pkl', 'rb'))
svd_model = pickle.load(open('svd_model.pkl', 'rb'))

# ---- Database management ----
import sqlite3

# Create connection
conn = sqlite3.connect('user_data.db')
c = conn.cursor()

# Create necessary tables
c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        user_id TEXT PRIMARY KEY,
        password TEXT
    )
''')
conn.commit()

# ---- DB Functions ----
# create a new account - sign up function
def add_userdata(user_id,password):
	c.execute('INSERT INTO users (user_id,password) VALUES (?,?)',(user_id,password))
	conn.commit()

# Function to check if user already exists in the database
def check_user_exists(user_id):
    c.execute('SELECT COUNT(*) FROM users WHERE user_id = ?', (user_id,))
    result = c.fetchone()
    return result[0] > 0  # If count is greater than 0, user exists

def login_user(user_id,password):
	c.execute('SELECT * FROM users WHERE user_id =? AND password = ?',(user_id,password))
	return c.fetchone()

# ---- Password security ----
import hashlib
def make_hashes(password):
	return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password,hashed_text):
	if make_hashes(password) == hashed_text:
		return hashed_text
	return False

# ---- App Setup (page configuration) ----
st.set_page_config(page_title="Hybrid Book Recommendation", layout="wide")

# ---- Session State Initialization ----
for key, default in {
    "page": "home",
    "authenticated": False,
    "selected_book": None,
    "pending_page": None,
    "logout_page": None
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ---- Page Navigation Functions ----
def go_to(page):
    st.session_state.pending_page = page
    st.rerun()

# ---- Navigation bar ----
def navigation():
    if st.session_state.authenticated:
        nav_options = ["Home", "Logout"]
        page_to_nav = {
            "home": "Home",
            "logout": "Logout"
        }
    else:
        nav_options = ["Home", "Login", "Sign Up"]
        page_to_nav = {
            "home": "Home",
            "login": "Login",
            "signup": "Sign Up"
        }
    
    current_nav = page_to_nav.get(st.session_state.page, "Home")
    
    selected = st.sidebar.selectbox(
        "Navigation",
        nav_options,
        index=nav_options.index(current_nav),
        key="nav_selectbox"
    )
    
    if st.session_state.pending_page:
        st.session_state.page = st.session_state.pending_page
        st.session_state.pending_page = None
    else:
        if selected == "Home":
            st.session_state.page = "home"
        elif selected == "Login":
            st.session_state.page = "login"
        elif selected == "Sign Up":
            st.session_state.page = "signup"
        elif selected == "Logout":
            st.session_state.page = "logout"

# ---- Login page ----
def login_page():
    st.header("Login Page")
    user_id = st.text_input("User ID")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        hashed_pswd = make_hashes(password)
        result = login_user(user_id,check_hashes(password,hashed_pswd))
        if result:
            st.session_state.authenticated = True
            st.session_state.user_id = user_id
            st.success(f"Welcome, {user_id}!")
            go_to("home")
        else:
            st.warning("Invalid credentials")

# ---- Sign-up page ----
def signup_page():
    st.header("Sign Up Page")
    new_userid = st.text_input("Enter User ID")
    new_password = st.text_input("Enter Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")

    if st.button("Create Account"):
        if new_password == confirm_password and new_userid:
            # Check if user already exists in the database
            if check_user_exists(new_userid):
                st.error("User ID already exists. Please choose a different ID.")
            else:
                add_userdata(new_userid, make_hashes(new_password))
                st.success(f"Account created for {new_userid}!")
                go_to("login")
        else:
            st.error("Passwords do not match or username is empty.")

# ---- Logout (Appear after login) ----
def logout_page():
    st.header("Logout")
    st.write("Are you sure you want to logout?")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Yes, Logout"): 
            st.session_state.authenticated = False
            st.session_state.user_id = None
            st.session_state.page = "home"
            st.rerun()
    with col2: 
        if st.button("Cancel"):
            go_to("home")
    
    # Add some styling
    st.markdown("""
    <style>
        .logout-container {
            max-width: 500px;
            margin: 0 auto;
            padding: 20px;
            text-align: center;
        }
        .logout-buttons {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-top: 30px;
        }
        .logout-message {
            font-size: 1.2rem;
            margin-bottom: 30px;
        }
    </style>
    """, unsafe_allow_html=True)

# ---- Recommender System Code ----
# ==== Popularity-based if the user is not login & not searching ====
def get_trending_books(top_n=10):
    C = cleaned_book_df['average_rating'].mean()
    m = cleaned_book_df['ratings_count'].quantile(0.9)
    trending_books = cleaned_book_df.loc[cleaned_book_df['ratings_count'] >= m].copy()

    def weighted_rating(x, m=m, C=C):
        v = x['ratings_count']
        R = x['average_rating']
        return (v/(v+m) * R) + (m/(m+v) * C)

    trending_books['score'] = trending_books.apply(weighted_rating, axis=1)
    trending_books = trending_books.sort_values('score', ascending=False)

    # Only pick the necessary columns from book_df
    book_info = book_df[['book_id', 'title', 'authors', 'publisher', 'description', 'url', 'average_rating']]
    # Merge and prioritize book_info (second df) columns
    trending_books_df = trending_books.drop(['title', 'authors', 'publisher', 'description', 'average_rating'], axis=1, errors='ignore')
    trending_books_df = trending_books_df.merge(book_info, on='book_id', how='left')
    return trending_books_df.head(top_n)

# ==== Hybrid recommender System (Content-based + Collaborative) ====
def hybrid_recommender(target_user_id, search_query, user_item_matrix, svd_model, n_recommendations=5, alpha=0.4, beta=0.3, gamma=0.3):
    # Content-based candidates
    direct_match_df, candidate_books = content_based_recommendation(search_query, top_n=100) # Retrieve more for the ease of sorting at following section
    if candidate_books.empty and direct_match_df.empty:
        print(f"No candidates found for query '{search_query}'")
        return pd.DataFrame()
    
    candidate_book_ids = candidate_books['book_id'].tolist()

    # Collaborative filtering scoring
    if target_user_id not in user_item_matrix.index or target_user_id == None:
        # New user --> cannot use collaborative, return content-based only
        print(f"User {target_user_id} not found in collaborative matrix. Returning content-based results.")
        return candidate_books.head(n_recommendations), direct_match_df.head(1)
    
    # --- User-based Recommendations ---
    user_recs = user_based_recommendations(target_user_id, candidate_books, user_item_matrix, n_books=n_recommendations*2)
    user_recs = user_recs[user_recs['book_id'].isin(candidate_book_ids)]

    # --- Item-based Recommendations ---
    item_recs = item_based_recommendations(target_user_id, candidate_books, user_item_matrix, n_books=n_recommendations*2)
    item_recs = item_recs[item_recs['book_id'].isin(candidate_book_ids)]

    # --- Model-based Recommendations (SVD) ---
    model_recs = model_based_recommendations(target_user_id, svd_model, user_item_matrix, candidate_book_ids, n_books=n_recommendations*2)

    # --- Normalize scores ---
    def normalize_scores(df, score_col):
        if df.empty:
            return df
        min_score = df[score_col].min()
        max_score = df[score_col].max()
        df[score_col] = (df[score_col] - min_score) / (max_score - min_score + 1e-10)
        return df

    user_recs = normalize_scores(user_recs, 'normalized_score')
    item_recs = normalize_scores(item_recs, 'item_score')
    model_recs = normalize_scores(model_recs, 'predicted_rating')

    # --- Create score dictionaries ---
    user_scores = dict(zip(user_recs['book_id'], user_recs['normalized_score']))
    item_scores = dict(zip(item_recs['book_id'], item_recs['item_score']))
    model_scores = dict(zip(model_recs['book_id'], model_recs['predicted_rating']))

    # --- Combine scores ---
    hybrid_scores = {}
    for book_id in candidate_book_ids:
        hybrid_scores[book_id] = (
            alpha * user_scores.get(book_id, 0) +
            beta * item_scores.get(book_id, 0) +
            gamma * model_scores.get(book_id, 0)
        )

    # --- Sort top final books ---
    top_books = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
    results = pd.DataFrame(top_books, columns=['book_id', 'hybrid_score'])

    # Add component scores for analysis
    results['user_score'] = results['book_id'].map(user_scores).fillna(0)
    results['item_score'] = results['book_id'].map(item_scores).fillna(0)
    results['model_score'] = results['book_id'].map(model_scores).fillna(0)

    # Merge with book titles
    book_info = book_df[['book_id', 'title', 'authors', 'publisher', 'description', 'url', 'average_rating']]
    results = results.drop(['title', 'authors', 'publisher', 'description', 'average_rating'], axis=1, errors='ignore')
    results = results.merge(book_info, on='book_id', how='left')

    return results, direct_match_df.head(1)

# ==== Content-based Recommender System ====
def content_based_recommendation(search_query, top_n=100):
    """
    Content-based recommendation with direct match checking.
    Returns:
    - direct_match_df: if any exact title match (found title in the dataset)
    - recommended_books_df: content-based recommendations (keyword recommendation)
    """

    # Step 1: Check if the query matches any title exactly
    direct_match = cleaned_book_df[cleaned_book_df['title'].str.lower() == search_query.strip().lower().replace(' ', '_')]
    
    if not direct_match.empty:
        # Found an exact match
        book_info = book_df[['book_id', 'title', 'authors', 'publisher', 'description', 'url', 'average_rating']]
        direct_match_df = direct_match.drop(['title', 'authors', 'publisher', 'description', 'average_rating'], axis=1, errors='ignore')
        direct_match_df = direct_match_df.merge(book_info, on='book_id', how='left')
    else:
        direct_match_df = pd.DataFrame()  # empty dataframe if no match

    # Step 2: Do normal content-based search
    query_embedding = bert_model.encode([search_query])
    distances, book_indices = bert_nn_model.kneighbors(query_embedding, n_neighbors=top_n + 1)

    book_indices = book_indices.flatten()
    distances = distances.flatten()
    similarities = 1 - distances

    # Step 3: Prepare result dataframe
    recommended_book_ids = index_to_book_id.iloc[book_indices].values
    recommended_books = pd.DataFrame({
        'book_id': recommended_book_ids,
        'similarity': similarities
    })

    # Merge to get full book details
    book_info = book_df[['book_id', 'title', 'authors', 'publisher', 'description', 'url', 'average_rating']]
    recommended_books_df = recommended_books.drop(['title', 'authors', 'publisher', 'description', 'average_rating'], axis=1, errors='ignore')
    recommended_books_df = recommended_books_df.merge(book_info, on='book_id', how='left')

    # Step 4: Remove direct match from recommendations (if it exists in the recommendations)
    if not direct_match_df.empty:
        recommended_books_df = recommended_books_df[~recommended_books_df['book_id'].isin(direct_match_df['book_id'])]

    return direct_match_df, recommended_books_df

# ==== Collaborative - User-based ====
def get_top_similar_users_knn(target_user_id, user_item_df):
    user_ids = user_item_df.index
    # Convert to sparse 
    user_item_matrix = csr_matrix(user_item_df.values)
    
    target_idx = user_ids.get_loc(target_user_id)
    distances, indices = user_knn.kneighbors(user_item_matrix[target_idx])
    
    # Return similar users (excluding the target user themselves)
    similar_users = user_ids[indices.flatten()[1:]]
    similarities = 1 - distances.flatten()[1:]  # Convert distances to similarities
    return pd.Series(similarities, index=similar_users)

def user_based_recommendations(target_user_id, content_based_df, user_item_matrix, n_books=5):
    # Get similar users from the user-based model
    similar_users = get_top_similar_users_knn(target_user_id, user_item_matrix)

    # Get books rated by similar users but not by target user
    target_books = set(cleaned_review_df[cleaned_review_df['user_id'] == target_user_id]['book_id'])
    similar_users_books = cleaned_review_df[cleaned_review_df['user_id'].isin(similar_users.index)]

    # Filter out books already rated by target user ==> Already read
    candidate_books = similar_users_books[~similar_users_books['book_id'].isin(target_books)]
    
    # Calculate weighted ratings based on user similarity
    candidate_books = candidate_books.copy()
    candidate_books['weighted_rating'] = candidate_books.apply(
        lambda x: x['rating'] * similar_users[x['user_id']], axis=1
    )

    # Get top rated books with weighted average
    book_scores = candidate_books.groupby('book_id').agg({
        'weighted_rating': 'sum',
        'user_id': 'count',
        'rating': 'mean'
    })

    # Normalize the weighted ratings by dividing by sum of similarities
    book_scores['similarity_sum'] = book_scores['user_id'].apply(
        lambda count: sum([similar_users[user] for user in 
                          candidate_books[candidate_books['book_id'] == book_scores.index[book_scores['user_id'] == count].values[0]]['user_id']])
    )
    
    book_scores['normalized_score'] = book_scores['weighted_rating'] / book_scores['similarity_sum']

    # Sort and get top books
    top_books = book_scores.sort_values('normalized_score', ascending=False).head(n_books)
    top_books = top_books.reset_index()

    # Merge with content-based results to filter out already recommended books
    book_info = content_based_df[['book_id', 'title', 'authors', 'publisher', 'description', 'url', 'average_rating']]
    recommended_books = top_books.drop(['title', 'authors', 'publisher', 'description', 'average_rating'], axis=1, errors='ignore')
    recommended_books = recommended_books.merge(book_info, on='book_id', how='left')

    return recommended_books

# ==== Collaborative - Item-based ====
def item_based_recommendations(target_user_id, content_based_df, user_item_matrix, n_books=5):
    # Get books rated by the target user
    target_books = cleaned_review_df[cleaned_review_df['user_id'] == target_user_id]['book_id'].values
    
    if len(target_books) == 0:
        return pd.DataFrame()  # Handle cold-start

    # Map book_id to matrix index
    book_to_idx = {book_id: idx for idx, book_id in enumerate(user_item_matrix.columns)}
    
    # For each book rated by the user, find similar books
    candidate_scores = defaultdict(float)
    for book_id in target_books:
        if book_id not in book_to_idx:
            continue
        book_idx = book_to_idx[book_id]
        similar_indices = item_indices[book_idx][1:]  # Exclude self
        similar_scores = item_similarities[book_idx][1:]
        
        # Aggregate scores across all similar items
        for sim_idx, sim_score in zip(similar_indices, similar_scores):
            similar_book_id = user_item_matrix.columns[sim_idx]
            candidate_scores[similar_book_id] += sim_score
    
    # Filter out books already rated by the user
    candidate_scores = {
        book_id: score for book_id, score in candidate_scores.items()
        if book_id not in target_books
    }
    
    # Sort and get top books
    top_books = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:n_books]
    top_books = pd.DataFrame(top_books, columns=['book_id', 'item_score'])
    
    # Merge with content-based results to filter out already recommended books
    book_info = content_based_df[['book_id', 'title', 'authors', 'publisher', 'description', 'url', 'average_rating']]
    recommended_books = top_books.drop(['title', 'authors', 'publisher', 'description', 'average_rating'], axis=1, errors='ignore')
    recommended_books = recommended_books.merge(book_info, on='book_id', how='left')

    return recommended_books

# ==== SVD Model - Predict Rating & Sort the result ==== 
def model_based_recommendations(target_user_id, svd_model, user_item_matrix, candidate_book_ids, n_books=5):
    # Get predictions for books in candidate_book_ids
    predictions = []
    for book_id in candidate_book_ids:
        try:
            pred_rating = svd_model.predict(target_user_id, book_id).est
            predictions.append((book_id, pred_rating))
        except:
            continue  # Ignore errors, no prediction for this book

    # Sort predictions by rating
    top_books = sorted(predictions, key=lambda x: x[1], reverse=True)[:n_books]
    top_book_ids = [book_id for book_id, _ in top_books]
    top_predicted_ratings = [pred for _, pred in top_books]

    # Create DataFrame for results
    results = pd.DataFrame({
        'book_id': top_book_ids,
        'predicted_rating': top_predicted_ratings
    })

    return results

# Scrap image 
import requests
from bs4 import BeautifulSoup
def get_image_url(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an error for bad status codes
        soup = BeautifulSoup(response.text, 'html.parser')

        # Try multiple selectors in case the class changes
        img_tag = (
            soup.find('img', class_='ResponsiveImage') or
            soup.find('img', {'role': 'presentation'}) or
            soup.select_one('.BookCover__image img')
        )

        if img_tag and 'src' in img_tag.attrs:
            return img_tag['src']
        else:
            return None

    except requests.RequestException as e:
        print(f"Error fetching the URL: {e}")
        return None

# ---- Main Home Page of the System ----
def home_page():
    # Custom CSS for styling
    st.markdown("""
    <style>
        body {
            background-color: black;
            color: white;
            font-family: Arial, sans-serif;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            text-align: center;
            margin-bottom: 20px;
        }
        .header h1 {
            font-size: 1.8em;
            margin: 0;
        }
        .search-result-card {
            background-color: #222;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            display: flex;
            gap: 20px;
            align-items: center;
        }
        .search-result-card img {
            width: 120px;
            height: 180px;
            object-fit: cover;
            border-radius: 5px;
        }
        .search-result-info {
            flex: 1;
        }
        /* Keep all your existing styles */
        .book-card {
            background-color: #333;
            padding: 20px;
            text-align: center;
            border-radius: 10px;
            min-height: 350px;
            transition: transform 0.3s;
        }
        .book-card:hover {
            transform: scale(1.05);
            box-shadow: 0 0 15px rgba(255, 255, 255, 0.2);
        }
        .book-card h3 {
            margin: 10px 0;
            font-size: 1.2em;
            height: 60px;
            overflow: hidden;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
        }
        .book-card p {
            margin: 5px 0;
            color: #ccc;
        }
        .book-card img {
            width: 120px;
            height: 180px;
            object-fit: cover;
            border-radius: 5px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            margin-bottom: 10px;
        }
    </style>
    """, unsafe_allow_html=True)

    # Container for the page
    st.markdown('<div class="container">', unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="header"><h1>Hybrid Based Book Recommendation System</h1></div>', unsafe_allow_html=True)

    # Search functionality
    search_term = st.text_input("Enter Book Title or Keyword", key="search_input")
    num_recommendations = 5
    # Slider
    with st.container():
        st.markdown('<div class="slider-container">', unsafe_allow_html=True)
        num_recommendations = st.slider(
                "Number of Recommendations",
                min_value=1,
                max_value=50,
                value=5,  # Default value
                key="num_rec_slider"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
    found_books = pd.DataFrame() # Store the direct match book record (title aspect)
    recommended_books = pd.DataFrame() # store all the recommendation

    if search_term:
        if (st.session_state.authenticated):
            recommended_books, found_books = hybrid_recommender(st.session_state.user_id, search_term, user_item_matrix, svd_model, n_recommendations=num_recommendations)
        else:
            found_books, recommended_books = content_based_recommendation(search_term, top_n=num_recommendations)

    # Display found book (if any)
    if not found_books.empty:
        book = found_books.iloc[0]  # Show first match
        st.markdown("### Found Book")
        # Display found book in a special card
        with st.container():
            st.markdown(f"""
            <div class="search-result-card">
                <img src="{get_image_url(book['url'])}" alt="{book['title']}">
                <div class="search-result-info">
                    <h2>{book['title']}</h2>
                    <p><em>by {book['authors']}</em></p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("View Details", key=f"found_{book['book_id']}"):
                st.session_state.selected_book = book
                go_to("book_details")
        
    st.markdown("---")
        
    st.markdown("### Recommended Books")
    if search_term:
        display_books = recommended_books
        if display_books.empty:
            st.warning("No books found matching your search. Showing trending books instead.")
            display_books = get_trending_books(num_recommendations)
    else:
        display_books = get_trending_books(num_recommendations)

    display_books = display_books.head(num_recommendations)
    
    # Create a 5-column grid for books
    cols = st.columns(5)
    for idx, book in display_books.iterrows():
        with cols[idx % 5]:
            book_url = book['url']
            book_title = book.get('title', 'Unknown Title')
            author = book.get('authors', 'Unknown Author')
            st.markdown(f"""
            <div class="book-card">
                <img src="{get_image_url(book_url)}" alt="{book_title}">
                <h3>{book_title}</h3>
                <p>by {author}</p>
            </div>
            """, unsafe_allow_html=True)

            if st.button("↪ View Details", key=f"book_{idx}"):
                st.session_state.selected_book = book
                go_to("book_details")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ---- Book details page ----
def book_details_page(book):
    st.markdown(f"""
    <style>
        .book-details-container {{
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
            color: white;
            background-color: #111;
            border-radius: 10px;
        }}
        .book-header {{
            display: flex;
            gap: 30px;
            margin-bottom: 30px;
        }}
        .book-cover {{
            flex: 0 0 300px;
        }}
        .book-cover img {{
            width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }}
        .book-info {{
            flex: 1;
        }}
        .book-title {{
            font-size: 2.2rem;
            margin-bottom: 10px;
            color: #fff;
        }}
        .book-author {{
            font-size: 1.5rem;
            color: #ccc;
            margin-bottom: 20px;
        }}
        .book-rating {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 20px;
        }}
        .stars {{
            color: gold;
            font-size: 1.2rem;
        }}
        .book-description {{
            line-height: 1.6;
            margin-bottom: 30px;
        }}
        .back-button {{
            margin-top: 20px;
            padding: 10px 20px;
            background-color: #333;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 1rem;
        }}
        .back-button:hover {{
            background-color: #444;
        }}
    </style>
    """, unsafe_allow_html=True)
    rating = book.get('average_rating', 'No person rated.')
    st.markdown(f"""
    <div class="book-details-container">
        <div class="book-header">
            <div class="book-cover">
                <img src="{get_image_url(book['url'])}" alt="{book['title']}">
            </div>
            <div class="book-info">
                <h1 class="book-title">{book['title']}</h1>
                <h2 class="book-author">by {book['authors']}</h2>
                <div class="book-rating">
                    <span>⭐ Rating: {rating}</span>
                </div>
                <div class="book-description">
                    <p>{book['description']}</p>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("← Back to Recommendations"):
        go_to("home")

# ---- Page Routing ----
if st.session_state.page == "home":
    home_page()
elif st.session_state.page == "login":
    login_page()
elif st.session_state.page == "signup":
    signup_page()
elif st.session_state.page == "book_details":
    if "selected_book" in st.session_state:
        book_details_page(st.session_state.selected_book)
    else:
        go_to("home")
elif st.session_state.page == "logout":
    logout_page()

# ---- Display navigation ----
navigation()