import code as cd
import streamlit as st
import textwrap

st.title("YouTube Assistant")

with st.form(key="my_form"):
    youtube_url = st.text_area(
        label="What is the Youtube video url?",
        max_chars = 50,
        key='youtube_url',
        height=10
    )
    query = st.text_area(
        label="Ask me something about the video",
        max_chars = 200,
        key="query",
        height=10
    )

    submit_button = st.form_submit_button(label='Submit')

if query and youtube_url:
    index, text_values = cd.create_embeddings(youtube_url)
    context = cd.perform_query_search(index, text_values, query, k=4)
    response = cd.generate_response_with_gpt(context)

    st.subheader('Answers')
    st.text_area(
        'Context from video',
        context,
        height=200)
    #st.text(youtube_url)
    st.text_area(
        'Generated response',
        response,
        height=500)
    # st.text(query)

    # index, text_values = create_embeddings(video_url)
# youtube_res = perform_query_search(index, text_values, 'what did the video say about tiktok?', 4)