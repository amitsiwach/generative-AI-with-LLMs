from langchain_community.llms import Bedrock
from langchain_aws import BedrockLLM
from langchain.chains import llm
from langchain.prompts import PromptTemplate
import boto3


#bedrock client
bedrock_client = boto3.client(
    service_name = "bedrock-runtime",
    region_name = "us-east-1"
)

model_id = "mistral.mistral-large-2402-v1:0"

llm = BedrockLLM(
    model_id = model_id,
    client = bedrock_client
)


def my_model(user_prompt):
    prompt_template = PromptTemplate(
        input_variables=['user_prompt'],
        template= "You are a chatbot. provide ans for {user_prompt}"
    )

    # Generate the prompt string using the template
    prompt_str = prompt_template.format(user_prompt=user_prompt)

    # Invoke the LLM with the prompt string
    response = llm.invoke(prompt_str)

    return response



user_prompt = "What is Purpose of human life?"
res = my_model(user_prompt)
print(res)