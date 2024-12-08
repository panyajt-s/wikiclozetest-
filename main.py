import streamlit as st
import pandas as pd
from openai import OpenAI
import os
import requests
from bs4 import BeautifulSoup
import json


st.sidebar.header("API Key Configuration")
api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY")
    )

    st.title("Cloze Test Generator from Wikipedia")
    url = st.text_input("Enter a Wikipedia URL:")

    if url and st.button("Generate Cloze Test"):
        try:
            response = requests.get(url)
            response.raise_for_status() 
            soup = BeautifulSoup(response.content, 'html.parser')
            paragraphs = soup.find_all('p')

            content = "\n".join(p.get_text() for p in paragraphs[:5])

            prompt_cloze_test = (
                "Generate a cloze test in JSON format with exactly 10 questions. Each test entry should include three keys: "
                "'question', 'answer', and 'explanation'. Return only a JSON array with no additional text or explanation. "
                "Here is the content for the cloze test:\n\n"
                f"{content}"
            )

            chat_completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": prompt_cloze_test}
                ]
            )

            output = chat_completion.choices[0].message.content.strip()

            try:
                if not output.startswith("[") or not output.endswith("]"):
                    output = output[output.find("["):output.rfind("]") + 1]

                json_output = json.loads(output)  
                cloze_test_df = pd.DataFrame(json_output)

                prompt_vocab = (
                    "From the following text, identify and extract up to 20 vocabulary words (nouns, verbs, and adjectives) along "
                    "with their definitions. Return the results in JSON format with each entry having keys: 'word', 'definition', and 'pos'.\n\n"
                    f"{content}"
                )

                vocab_completion = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "user", "content": prompt_vocab}
                    ]
                )

                vocab_output = vocab_completion.choices[0].message.content.strip()

                try:
                    if not vocab_output.startswith("[") or not vocab_output.endswith("]"):
                        vocab_output = vocab_output[vocab_output.find("["):vocab_output.rfind("]") + 1]

                    vocab_json_output = json.loads(vocab_output)  
                    vocab_df = pd.DataFrame(vocab_json_output)

                    cloze_test_df.index = range(1, len(cloze_test_df) + 1)  
                    vocab_df.index = range(1, len(vocab_df) + 1) 

                    st.subheader("Generated Cloze Test")
                    st.dataframe(cloze_test_df, use_container_width=True)

                    csv_cloze = cloze_test_df.to_csv(index=False)
                    st.download_button(
                        label="Download Cloze Test as CSV",
                        data=csv_cloze,
                        file_name="cloze_test.csv",
                        mime="text/csv"
                    )

                    st.subheader("Interesting Vocabulary")
                    st.dataframe(vocab_df, use_container_width=True)

                    csv_vocab = vocab_df.to_csv(index=False)
                    st.download_button(
                        label="Download Vocabulary as CSV",
                        data=csv_vocab,
                        file_name="vocabulary.csv",
                        mime="text/csv"
                    )

                except ValueError:
                    st.error("The response for vocabulary wasn't valid JSON. Please try again or check the API response.")

            except ValueError:
                st.error("The response from the API wasn't valid JSON. Please try again or check the API response.")

        except requests.exceptions.RequestException as e:
            st.error(f"An error occurred while fetching the URL: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
else:
    st.warning("Please enter your OpenAI API key in the sidebar.")
