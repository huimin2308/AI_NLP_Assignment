What to do before using the application:
1. check .env file exists in the folder or not
2. If not, create an account at OpenRouter: https://openrouter.ai/ 
3. Then, create API Keys
4. Create .env file, and write the below sentence: (Replace the <<API KEYS>> with yours)
OPENROUTER_API_KEY=<<API KEYS>>

Before run the program, type commands below at the terminal:
1. Download relevant library
# pip install -r requirement.txt

2. Download nltk needed package
# python text_preprocessing.py

After importing and downloading relevant library, you can now run the program:
# streamlit run app.py