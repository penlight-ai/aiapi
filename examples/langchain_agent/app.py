import typing
import os
import asyncio
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import HumanMessage
from aiser import RestAiServer, ServerAgent
from aiser.models import ChatMessage


class CustomCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self._tokens: typing.List[str] = []
        self._has_finished: bool = False

    def on_llm_new_token(self, token: str, **kwargs: typing.Any) -> None:
        self._tokens.append(token)

    def on_llm_end(self, response, **kwargs: typing.Any) -> None:
        self._has_finished = True

    async def get_generated_tokens(self) -> typing.AsyncGenerator[typing.List[str], None]:
        while (not self._has_finished) or (len(self._tokens) > 0):
            if len(self._tokens) > 0:
                yield self._tokens
                self._tokens = []
            await asyncio.sleep(0.05)


class PromptBasedAgent(ServerAgent):
    def __init__(
            self,
            agent_id: str,
            starting_prompt: str,
            openai_api_key: str,
            model_name: str,
    ):
        super().__init__(agent_id=agent_id)
        self._starting_prompt = starting_prompt
        self._openai_api_key = openai_api_key
        self._model_name = model_name

    async def reply(self, messages: typing.List[ChatMessage]) -> typing.AsyncGenerator[ChatMessage, None]:
        callback_handler = CustomCallbackHandler()
        ai_model = self._make_model(
            callback_handler=callback_handler
        )
        model_input = self._make_model_input(messages=messages)
        model_execution_task = asyncio.create_task(self._execute_model(
            chat_model=ai_model,
            model_input=model_input
        ))

        async for tokens in callback_handler.get_generated_tokens():
            for token in tokens:
                yield ChatMessage(
                    text_content=token,
                )
        await model_execution_task

    def _make_model(self, callback_handler: CustomCallbackHandler) -> ChatOpenAI:
        return ChatOpenAI(
            openai_api_key=self._openai_api_key,
            model=self._model_name,
            streaming=True,
            temperature=0,
            callbacks=[callback_handler],
            verbose=True
        )

    def _make_model_input(self, messages: typing.List[ChatMessage]) -> str:
        starting_message = ChatMessage(
            text_content=self._starting_prompt
        )
        messages_combined = [starting_message] + list(messages)
        return self._messages_to_single_str(messages_combined)

    def _messages_to_single_str(self, messages: typing.List[ChatMessage]) -> str:
        final_prompt = ""
        for i, message in enumerate(messages):
            final_prompt += f"message {i}: {message.text_content}\n"
        final_prompt += f"message {len(messages)}:"
        return final_prompt

    async def _execute_model(self, chat_model, model_input: str):
        await chat_model.apredict_messages(messages=[
            HumanMessage(content=model_input)
        ])


if __name__ == '__main__':
    # Loading Environment Variables:
    # Duplicate the file sample.env and rename it to .env then fill in the values in the .env file.
    # Set the environment variable AISER_ENVIRONMENT to "Development" only for local development.
    # Otherwise, you can leave it empty or set it to "Production"
    from dotenv import load_dotenv

    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    opposites_agent = PromptBasedAgent(
        agent_id='8d4ac909-ce3e-48c6-8233-63fa21da5b23',  # replace this with your agent id
        starting_prompt='''
                You are Opposite Agent. Whenever you receive a message from the user, 
                you should reply with the opposite of that message.
                Don't reply with anything other than the exact opposite of the user's message.
                ''',
        openai_api_key=OPENAI_API_KEY,
        model_name='gpt-3.5-turbo'
    )

    general_assistant_agent = PromptBasedAgent(
        agent_id='4b4b4bad-bdde-4cba-9170-ac8ae11d994a',  # replace this with your agent id
        starting_prompt='''
            You are a friendly and helpful AI assistant.
            ''',
        openai_api_key=OPENAI_API_KEY,
        model_name='gpt-3.5-turbo'
    )

    server = RestAiServer(
        agents=[
            opposites_agent,
            general_assistant_agent
        ],
        port=5000
    )
    server.run()
