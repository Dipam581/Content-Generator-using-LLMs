import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain_huggingface import HuggingFaceEndpoint

# Get response from LLama 2
def get_LLAma_response(text, number, style):
    config = {'max_new_tokens': 256,'temperature' : 0.8, 'repetition_penalty': 1.1}
    llm = CTransformers(model='llama-2-7b-chat.ggmlv3.q8_0.bin',model_type='llama', config=config)
    # repo_id = "meta-llama/Meta-Llama-3-8B"
    # llm = HuggingFaceEndpoint(
    #     repo_id=repo_id, max_length=128, temperature=0.8, token="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"
    # )

    template = '''
    Write a blog for {style} job profile for a topic {text} within {number} words.
    '''

    prompt = PromptTemplate(input_variables=["style", "text", "number"], template=template)

    return llm(prompt.format(style=style, text=text, number=number))

st.set_page_config(page_title="Generate Blogs",
                   page_icon="😂",
                   layout="centered",
                   initial_sidebar_state="collapsed")

st.header("Generate blogs.......")

input_text = st.text_input("Enter the Blog topic")

col1, col2 = st.columns([5, 5])

with col1:
    no_of_words = st.text_input("No of Words", value=None)

with col2:
    blog_style = st.selectbox('Writing the blog for', ('Researchers', 'Data Scientist', ' Normal People'), index=0)

submit = st.button("Generate")

if submit:
    response = get_LLAma_response(input_text, no_of_words, blog_style)
    st.write(response)
