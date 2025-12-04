import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space

# Set page configuration
st.set_page_config(page_title="Interact with your PDF") # Later change name

# Set the sidebar content
with st.sidebar:
    st.title('Chat History')
    st.markdown('''
    Pending add content
    ''')

    add_vertical_space(4)
    st.write('Testing')


def main():
    load_dotenv()

    

    st.header("VK: CSV Based Chatbot! Ask Your Csv📈")

    csv_file = st.file_uploader("Upload a CSV file", type="csv")
    if csv_file is not None:

        agent = create_csv_agent(
            OpenAI(model_name="gpt-3.5-turbo-instruct"), csv_file, verbose=True)

        user_question = st.text_input("Ask a question about your CSV: ")

        if user_question is not None and user_question != "":
            with st.spinner(text="In progress..."):
                st.write(agent.run(user_question))


if __name__ == "__main__":
     main()