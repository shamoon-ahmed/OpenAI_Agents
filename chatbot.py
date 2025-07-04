import os
import chainlit as cl
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig
from dotenv import load_dotenv
import datetime

load_dotenv()

API_KEY=os.getenv('GEMINI_API_KEY')

current_date = datetime.date.today()

provider = AsyncOpenAI(
    api_key=API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

model = OpenAIChatCompletionsModel(
    model="gemini-1.5-flash",
    openai_client=provider,
    )

config = RunConfig(
    model_provider=provider,
    model=model,
    tracing_disabled=True
    )

agent = Agent(
        name="Nice Sip Water Supplier",
        instructions=f'''
            You are a sales assistant for Nice Sip, a mineral water supplier in Karachi.
            We sell 19L (Rs.70), 13L (Rs.40), and 5L (Rs.30) bottles.
            Only deliver within Karachi on Monday, Wednesday and Friday. Based on customers' query, Ask the customer:
                What bottle size(s) they want
                How many bottles they need
                Whether delivery should be daily or weekly (no monthly orders allowed)
                Their delivery address
                Preferred payment schedule (daily, weekly, or monthly)
            But don't ask these questions at once. Be precise and don't be overwhelming.
            Remember the current date which is: {current_date}
            Once the order is confirmed, generate a structured bill with customer's name, no. of bottles and size, address, Order placed on, estimated delivery time (depending on the coming delivery day eg: mon, tu or wed)
            Be friendly and clear. Politely reject orders outside Karachi or small monthly orders.
        ''',
        model=model,
        )

# this is just displaying a beginning message from the ai (the very first message)
@cl.on_chat_start
async def start():

    cl.user_session.set("history", [])

    await cl.Message(content="Hey! We're Nice Sip Mineral Water Suppliers. How can we help you today?").send()

# this is when the conversation starts
@cl.on_message
async def chat(message: cl.Message):

    history = cl.user_session.get("history")

    history.append({"role":"user", "content":message.content})

    result = await Runner.run(
        agent,
        input=history,
        run_config=config)
    
    history.append({"role":"assistant", "content":result.final_output})

    cl.user_session.set("history", history)

    await cl.Message(content=result.final_output).send()

