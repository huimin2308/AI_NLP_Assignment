import streamlit as st
import streamlit.components.v1 as components
from streamlit_extras.stylable_container import stylable_container

# Set page configuration
st.set_page_config(page_title="Hybrid Book Recommendation", layout="wide")

if "page" not in st.session_state:
    st.session_state.page = "home"
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "selected_book" not in st.session_state: 
    st.session_state.selected_book = None
if "pending_page" not in st.session_state:
    st.session_state.pending_page = None
if "logout_page" not in st.session_state:
    st.session_state.logout_page = None

# Sample Book data
# return出来的 dataset放这里
# The functions mainly use dataset 
# e.g. book_data = ?
book_data = [
    {"book_id": "287141", "title": "The Silent Patient", "author": "Alex Michaelides", "image_url": "https://images-na.ssl-images-amazon.com/images/I/81L5Lor1fRL._AC_UL600_SR600,600_.jpg"},
    {"book_id": "287142", "title": "Where the Crawdads Sing", "author": "Delia Owens", "image_url": "https://images-na.ssl-images-amazon.com/images/I/81O1oy0y9eL._AC_UL600_SR600,600_.jpg"},
    {"book_id": "287143", "title": "Atomic Habits", "author": "James Clear", "image_url": "https://images-na.ssl-images-amazon.com/images/I/81bGKUa1e0L._AC_UL600_SR600,600_.jpg"},
    {"book_id": "287144", "title": "It Ends with Us", "author": "Colleen Hoover", "image_url": "https://images-na.ssl-images-amazon.com/images/I/71PNGYHykrL._AC_UL600_SR600,600_.jpg"},
    {"book_id": "287145", "title": "The Midnight Library", "author": "Matt Haig", "image_url": "https://images-na.ssl-images-amazon.com/images/I/81YzHKeWq7L._AC_UL600_SR600,600_.jpg"},
    {"book_id": "287146", "title": "Educated", "author": "Tara Westover", "image_url": "https://images-na.ssl-images-amazon.com/images/I/71y9ZfX6YBL._AC_UL600_SR600,600_.jpg"},
    {"book_id": "287147", "title": "The Alchemist", "author": "Paulo Coelho", "image_url": "https://images-na.ssl-images-amazon.com/images/I/71aFt4+OTOL._AC_UL600_SR600,600_.jpg"},
    {"book_id": "287148", "title": "Dune", "author": "Frank Herbert", "image_url": "https://images-na.ssl-images-amazon.com/images/I/81ym3QUd3KL._AC_UL600_SR600,600_.jpg"},
    {"book_id": "287149", "title": "Project Hail Mary", "author": "Andy Weir", "image_url": "https://images-na.ssl-images-amazon.com/images/I/91XSPJLmYQL._AC_UL600_SR600,600_.jpg"},
    {"book_id": "287150", "title": "The Seven Husbands of Evelyn Hugo", "author": "Taylor Jenkins Reid", "image_url": "https://images-na.ssl-images-amazon.com/images/I/91Oth5WxVQL._AC_UL600_SR600,600_.jpg"}
]

# Function to change page
def go_to(page):
    st.session_state.pending_page = page
    st.rerun()

# Navigation bar
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

# Login page
def login_page():
    st.header("Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if username == "admin" and password == "1234":
            st.session_state.authenticated = True
            st.success("Welcome back, admin!")
            go_to("home")
        else:
            st.error("Invalid credentials")

# Sign-up page
def signup_page():
    st.header("Sign Up Page")
    new_username = st.text_input("Choose a Username")
    new_password = st.text_input("Choose a Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")

    if st.button("Create Account"):
        if new_password == confirm_password and new_username:
            st.success(f"Account created for {new_username}!")
            go_to("login")
        else:
            st.error("Passwords do not match or username is empty.")


# Logout only appear after login
def logout_page():
    st.header("Logout")
    st.write("Are you sure you want to logout?")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Yes, Logout"): # Perform return normal dataset to book_data here
            st.session_state.authenticated = False
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
    

# Book details page
# User click the book, will navigate to this screen
# RMD add rating and description
# put rating {book['rating']}
# put desription {book['description']} in <p>
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
    
    st.markdown(f"""
    <div class="book-details-container">
        <div class="book-header">
            <div class="book-cover">
                <img src="{book['image_url']}" alt="{book['title']}">
            </div>
            <div class="book-info">
                <h1 class="book-title">{book['title']}</h1>
                <h2 class="book-author">by {book['author']}</h2>
                <div class="book-rating">
                    <span>⭐ Rating: 4.35</span> 
                </div>
                <div class="book-description">  
                    <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.</p>
                    <p>Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.</p>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("← Back to Recommendations"):
        go_to("home")

        
# Home page with book recommendations
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

    num_recommendations = 50
    
    # Store user input data
    search_term = st.text_input("Enter Title or Author", key="search_input")
    
    found_books = [] # --> use to store match books (bert embedding)
    recommended_books = [] # --> use to store array data after Content-Based Filtering

    # Check user input for search
    if search_term:
        # Search in titles and authors
        # Store the return match books
        found_books = [book for book in book_data 
                      if search_term.lower() in book['title'].lower() 
                      or search_term.lower() in book['author'].lower()]
        
        # Recommendation logic
        # If found books,
        if found_books:
            # If books found, recommend books by same author
            # Store the Content-Based Filtering books
            recommended_books = [book for book in book_data 
                               if book['author'].lower() == found_books[0]['author'].lower() 
                               and book['book_id'] != found_books[0]['book_id']][:5]
        else:
            # If no books found, recommend books with similar words
            # If no books found, store the Content-Based Filtering books
            recommended_books = [book for book in book_data 
                               if any(word in book['title'].lower() 
                                     for word in search_term.lower().split())][:5]
        
    
    # Display found book
    if found_books:
        st.markdown("### Found Book")
        book = found_books[0]  # Show first match
        
        # Display found book in a book card
        with st.container():
            st.markdown(f"""
            <div class="search-result-card">
                <img src="{book['image_url']}" alt="{book['title']}">
                <div class="search-result-info">
                    <h2>{book['title']}</h2>
                    <p><em>by {book['author']}</em></p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("View Details", key=f"found_{book['book_id']}"):
                st.session_state.selected_book = book
                go_to("book_details")
        
        st.markdown("---")
        
        
    # Slider for number of recommendations
    if search_term:
        st.markdown("### Recommended Books")
        
        with st.container():
            st.markdown('<div class="slider-container">', unsafe_allow_html=True)
            num_recommendations = st.slider(
                "Number of recommended books",
                min_value=1,
                max_value=50,
                value=10,  # Default value
                key="num_rec_slider"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
        recommended_books = recommended_books[:num_recommendations]
            
    # Display recommended books after search 
    # If search_term is not empty, assign recommended_books to display_book, 如果user search book就show recommended book
    # If serach_term is empty, assign book_data to display_book (book_data = dataset at above), login一进来就show book_data
    display_books = recommended_books if search_term else book_data[:num_recommendations]
    
    if not display_books and search_term:
        st.warning("No books found matching your search. Showing popular books instead.")
        display_books = book_data[:num_recommendations]
    
    # Create a 5-column grid for books
    cols = st.columns(5)
    for idx, book in enumerate(display_books):
        with cols[idx % 5]:
            st.markdown(f"""
            <div class="book-card">
                <img src="{book['image_url']}" alt="{book['title']}">
                <h3>{book['title']}</h3>
                <p>by {book['author']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Add View Details button with custom styling
            with stylable_container(
                key=f"custom_button_container_{idx}",
                css_styles="""
                    button {
                        background-color: none;
                        border: none;
                        outline: none;
                        color: yellow;
                        font-weight: bold;
                        font-style: italic;
                    }
                """
            ):
                if st.button("↪ View Details", key=f"book_{idx}", type="tertiary"):
                    st.session_state.selected_book = book
                    go_to("book_details")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Routing logic
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
    

# Display navigation
navigation()