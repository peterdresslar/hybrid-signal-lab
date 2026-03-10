import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import asyncio

    return asyncio, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Signaling

    We explore the ways in which we might detect signal from one LLM instance, and ways that signals might be distilled and reflected back into that instance.

    ## Chatbot
    This front-end code is based in part on Marimo's `streaming_custom.py` example.
    """)
    return


@app.cell
def _(asyncio, mo):
    async def streaming_echo_model(messages, config):
        """This chatbot echoes what the user says, word by word.

        Yields individual delta chunks that are accumulated by marimo.
        This follows the standard streaming pattern used by OpenAI, Anthropic,
        and other AI providers.
        """
        # Get the user's message
        user_message = messages[-1].content

        # Stream the response word by word
        response = f"You said: '{user_message}'. Here's my response streaming word by word!"
        words = response.split()

        for word in words:
            yield word + " "  # Yield delta chunks
            await asyncio.sleep(0.2)  # Delay to make streaming visible

    chatbot = mo.ui.chat(
        streaming_echo_model,
        prompts=["Hello", "Tell me a story", "What is streaming?"],
        show_configuration_controls=True
    )
    return (chatbot,)


@app.cell
def _(chatbot):
    chatbot
    return


if __name__ == "__main__":
    app.run()
