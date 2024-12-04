import openai
from dotenv import load_dotenv
import streamlit as st
import guardrails as gd
from guardrails.guard import Guard
import re
import os


# model = ChatOpenAI()

# Now you can access the API key in your code
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

def without_guardrails(text):
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Translate the " + text + " to English"}
    ],
    max_tokens=2048,
    temperature=0
)
    result=response['choices'][0]['message']['content']
    # values_in_quotes = re.findall(r'"(.*?)"', result)
    # return values_in_quotes[1]
    return result

rail_str = """
<rail version="0.1">

    <script language='python'>
    from profanity_check import predict
    from guardrails.validator_base import Validator, register_validator

    @register_validator(name="is-profanity-free", data_type="string")
    class IsProfanityFree(Validator):
        global predict
        global EventDetails
        def validate(self, key, value, schema) -> Dict:
            text = value
            prediction = predict([value])
            if prediction[0] == 1:
                raise EventDetail(
                    key,
                    value,
                    schema,
                    f"Value {value} contains profanity language"
                )                
            return schema
    </script>        

    <output>
        <string
            name="translated_statement"
            description="Translate the given statement into english"
            format="is-profanity-free"
            on-fail-is-profanity-free="fix" />
    </output>

    <prompt>
        Translate the given statement into english:
        {{statement_to_be_translated}}
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
            without_guardrails_result=without_guardrails(text_area)
            st.success(without_guardrails_result)
            st.warning("Translation with Guardrails")

            raw_llm_response, validated_response = guard(
                openai.ChatCompletion.create,
                model="gpt-3.5-turbo",
                messages = [
                    {"role": "user", "content": "Translate the statement into English: {text_area}."}
                    ],
                max_tokens=2048,
                temperature=0)

            # messages = [
            #     {"role": "user", "content": f"Translate the statement into English: {text_area}"}
            # ]

            # raw_response = openai.ChatCompletion.create(
            #     model="gpt-3.5-turbo",
            #     messages=messages,
            # )
            # # Extract the content from the OpenAI response
            # raw_text = raw_response['choices'][0]['message']['content']
            

            # # Validate the response using Guard
            # raw_llm_response, validated_response = guard(prompt_params={"statement_to_be_translated": text_area})



            # Now call the guard with the messages
            # raw_llm_response, validated_response = guard(messages=messages)


            st.success(f"Validated Output: {validated_response}")

if __name__ == "__main__":
    main()