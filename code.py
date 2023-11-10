import os
from dotenv import load_dotenv
import re
from youtube_transcript_api import YouTubeTranscriptApi
import numpy as np
import pandas as pd
import nltk

from sentence_transformers import SentenceTransformer
import faiss
from nltk.tokenize import sent_tokenize 
import openai

# load .env variables
load_dotenv()

# Get the OpenAI API key from .env
openai_api_key = os.environ.get("OPENAI_API_KEY")

"""
Get video_id from video_url using regex
"""
def get_video_id(video_url):
    youtube_id_match = re.search(r'(?<=v=)[^&#]+', video_url)
    youtube_id_match = youtube_id_match or re.search(r'(?<=be/)[^&#]+', video_url)
    if youtube_id_match:
        return youtube_id_match.group(0)
    else:
        return None

"""
Generate response with gpt3.5 ChatCompletion
"""
def generate_response_with_gpt(context):
    openai.api_key = openai_api_key
    response = openai.ChatCompletion.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": "You are a highly efficient virtual assistant designed to extract valuable information from video transcripts. Your primary goal is to provide insightful and concise summaries of the content within the transcripts. You excel in identifying key topics, extracting relevant details, and presenting the information in a clear and coherent manner. Your users rely on you to distill complex video content into easily understandable insights. Keep in mind the importance of accuracy, clarity, and brevity in your responses."},
            {"role": "user", "content": f"Answer the question based on the context given: {context}"},
        ],
        temperature=0
    )
    return response['choices'][0]['message']['content']

"""
This is to perform similarity search 
1. Builds vector db, create embeddings
2. Create embeddings with IndexFlatL2
3. Search and retrieve by comparing query with nearest k, default to 4
4. call chatcompletion

Will split by sentence. not word. 
Because i need to find specific sentences that are most similar to query. 
This could be useful for this app, which needs semantic search for Q&A
"""

def create_embeddings(video_url):
    video_id = get_video_id(video_url)
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    text_values = [text['text'] for text in transcript]

    embeddings = []
    for text in text_values:
        openai.api_key = openai_api_key
        response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
        embeddings.append(response['data'][0]['embedding'])
    sentence_embeddings = np.array(embeddings)

    index = faiss.IndexFlatL2(sentence_embeddings.shape[1])
    index.add(sentence_embeddings)

    return index, text_values

def perform_query_search(index, text_values, query, k):
    openai.api_key = openai_api_key
    response = openai.Embedding.create(input=query, model="text-embedding-ada-002")
    #embeddings = response['data'][0]['embedding']
    k=10
    xq = np.array([response['data'][0]['embedding']])
    D, I = index.search(xq, k) # index is from above

    responses = [text_values[i] for i in I[0]] # text values are from above
    context = ' '.join(responses)
    return context

"""
Previous code for reference
"""
# index, text_values = create_embeddings(video_url)
# youtube_res = perform_query_search(index, text_values, 'what did the video say about tiktok?', 4)
# print(youtube_res) # this prints none. Why!!!! 


# def create_vector_and_query(video_url, query, k):
#     # make transcript
#     video_id = get_video_id(video_url)
#     print(video_id)
#     transcript = YouTubeTranscriptApi.get_transcript(video_id)
#     text_values = [text['text'] for text in transcript] # index by sentence

#     # model = SentenceTransformer('bert-base-nli-mean-tokens')
#     # sentence_embeddings = model.encode(text_values)
#     # print(sentence_embeddings.shape) # returns (1415,768)

#      # create embeddings with openAI text-embedding-ada-002
#     embeddings = []
    
#     for text in text_values:
#         openai.api_key = openai_api_key
#         response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
#         embeddings.append(response['data'][0]['embedding'])
#     sentence_embeddings = np.array(embeddings)
#     print(sentence_embeddings.shape)

#     # build FAISS vector db with embeddings
#     index = faiss.IndexFlatL2(sentence_embeddings.shape[1])
#     #print(index.is_trained) returns true
#     index.add(sentence_embeddings)
#     #print(index.ntotal) returns 1415

#     # k = k
#     # xq = model.encode([query])
#     # D, I = index.search(xq, k) # this is search
#     # print (I) # returns [[1220 1261 1239 1247]], which has similar meaning to query
#     #  # return the actual sentences instead of their indices
#     # for i in I[0]:
#     #     print(text_values[i])
    
#     # create search vector with openAI embedding
#     response = openai.Embedding.create(input=query, model="text-embedding-ada-002")
#     xq = np.array([response['data'][0]['embedding']])
#     D, I = index.search(xq, k) # this is search
#     # return the actual sentences instead of their indices
#     responses = []
#     for i in I[0]:
#         responses.append(text_values[i])

#     # Join all responses into a single string
#     context = ' '.join(responses)
#     return context

#     # Call the chat function with the context
#     chat_response = chat(context)
#     return chat_response

#create_vector_and_query(video_id, 'what did they say about ransomware?', 10)


# def get_response_from_query(db, query, k=4):
#     # text-davinci can handle 4097 tokens

#     docs = db.similarity_search(query, k=k) # what's wrong?
#     docs_page_content = " ".join([d.page_content for d in docs])
#     # print(docs_page_content)
    
#     llm = OpenAI(model="text-davinci-003")

#     prompt = PromptTemplate(
#         input_variables=["question", "docs"],
#         template="""
#         You are a helpful YouTube assistant that can answer questions about videos based on video's transcript.
#         Answer the following question: {question}
#         By searching the following video transcript: {docs}
#         Only use the factual information from the transcript to answer the question.
#         If you feel like you don't have enough information to answer the question, say "I don't know".
#         Your answers should be detailed.
#         """
#     )

#     chain = LLMChain(llm=llm, prompt=prompt)

#     response = chain.run(question=query, docs=docs_page_content)
#     # print(response)
#     # reformating
#     response = response.replace("\n", "")

# print(get_response_from_query(db,"What does it say about ransomeware"))