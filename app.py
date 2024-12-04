import openai
from dotenv import load_dotenv
import streamlit as st
import guardrails as gd
from guardrails.guard import Guard
from profanity_check import predict
from guardrails.validator_base import Validator, register_validator
import re
import os


# model = ChatOpenAI()

# Now you can access the API key in your code
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

def my_llm_api(**kwargs) -> str:
    """Custom LLM API wrapper for GPT-3.5-turbo.

    At least one of `messages` should be provided.

    Args:
        **kwargs: Any additional arguments to be passed to the LLM API.

    Returns:
        str: The output of the LLM API.
    """
    # Extract messages from kwargs or set to an empty list if not provided
    messages = kwargs.pop("messages", [])
    
    # Perform optional message preprocessing
    updated_messages = some_message_processing(messages)
    
    # Ensure model is provided, default to "gpt-3.5-turbo" if not
    model = kwargs.pop("model", "gpt-3.5-turbo")

    # Call OpenAI ChatCompletion API
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=updated_messages,
            **kwargs  # Pass additional arguments (e.g., temperature, max_tokens)
        )
        # Extract the text content from the response
        llm_output = response["choices"][0]["message"]["content"]
        values_in_quotes = re.findall(r'"(.*?)"', llm_output)
        return values_in_quotes[1]
    except Exception as e:
        raise RuntimeError(f"Failed to call GPT-3.5-turbo API: {e}")

    return llm_output

def some_message_processing(messages):
    """Optional message preprocessing logic.

    Args:
        messages (list): The list of messages for the LLM.

    Returns:
        list: The processed list of messages.
    """
    # Add any custom processing logic here, if needed.
    # For example, ensure all messages follow the OpenAI API structure:
    return [{"role": msg["role"], "content": msg["content"]} for msg in messages]


@register_validator(name="is-profanity-free", data_type="string")
class IsProfanityFree(Validator):
    global predict
    
    def validate(self, value, **kwargs):

        prediction = predict([value])
        if prediction[0] == 1:
            raise ValueError(f"The input '{value}' contains profanity.")
        return value
    
# Define a Rail schema (XML format) for Guardrails
rail_str = """
<rail version="0.1">   

<output>
    <string
        name="translated_statement"
        description="Translate the given statement into english language"
        format="is-profanity-free"
        on-fail-is-profanity-free="fix" 
    />
</output>

<prompt>

Check the following text for profanity and ensure it is clean:

{{text_area}}

@complete_json_suffix
</prompt>

</rail>
"""
guard = Guard.from_rail_string(rail_str)

def main():
    st.title("Gardrails Implementation in LLMs")

    text_area = st.text_area("Enter the text to translate")

    if st.button("Translate"):
        if len(text_area) >0:
            st.info(text_area)

            st.warning("Without Guardrails")
            # without_guardrails_result=without_guardrails(text_area)
            messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Translate the " + text_area + " to English"}
                ]
            result = my_llm_api(messages=messages, temperature=0.7, max_tokens=100)
                    
            st.success(result)
            st.warning("Translation with Guardrails")
            # raw_llm_response, validated_response = guard(
            #     openai.ChatCompletion.create,
            #     model="gpt-3.5-turbo",
            #     messages = [
            #         {"role": "user", "content": "Translate the statement into English: {text_area}."}
            #         ],
            #     max_tokens=2048,
            #     temperature=0)
            
            # validated_response = guard(
            #     openai.ChatCompletion.create,
            #     messages = [
            #         {"role": "user", "content": f"Translate the statement into English: {text_area}."}
            #         ],
            #         model="gpt-3.5-turbo"
            # )
            # validated_response = guard(
            #     openai.ChatCompletion.create,
            #     prompt_params={"statement_to_be_translated": text_area},
            #     model="gpt-3.5-turbo"
            # )

            # messages = [
            #         {"role": "user", "content": "Translate the statement into English: {text_area}."}
            #         ]

            # # Wrap your LLM API call
            # validated_response = guard(
            #     my_llm_api,
            #     messages=messages,
            #     temperature=0.7, 
            #     max_tokens=100
            # )

            # print(f"{result.validated_output}")

            validated_response = guard(
                my_llm_api,
                messages=messages,
                model="gpt-3.5-turbo")

            # response = openai.ChatCompletion.create(
            #     model="gpt-3.5-turbo",
            #     messages=messages
            # )

            # response1 = my_llm_api(messages=messages, temperature=0.7, max_tokens=100)

            # # st.success(response)
            # st.success(response1)

            st.success(f"Validated Output:, {validated_response}")

if __name__ == "__main__":
    main()