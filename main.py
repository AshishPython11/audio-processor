from __future__ import annotations

from typing import AsyncIterable
import asyncio
from dotenv import load_dotenv
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm, tokenize
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import cartesia, deepgram, openai, silero
import requests
load_dotenv()


async def entrypoint(ctx: JobContext):
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            "You are a voice assistant created by LiveKit. Your interface with users will be voice. "
            "You should use short and concise responses, and avoiding usage of unpronouncable punctuation."
        ),
    )

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    def _before_tts_cb(agent: VoicePipelineAgent, text: str | AsyncIterable[str]):
        # The TTS is incorrectly pronouncing "LiveKit", so we'll replace it with a phonetic
        # spelling
        # Estimate audio length based on text length
        if isinstance(text, str):
            words = text.split()
            num_words = len(words)
            # This is a very rough estimate, assuming ~0.5 seconds per word
            estimated_length_seconds = num_words * 0.5

            print(
                f"Estimated audio length for '{text}': {estimated_length_seconds:.2f} seconds"
            )
        elif isinstance(text, AsyncIterable):

            async def async_len(iterable):
                count = 0
                async for _ in iterable:
                    count += 1
                return count

            async def handle_async_text():
                total_words = 0
                async for chunk in text:
                    words = chunk.split()
                    total_words += len(words)
                estimated_length_seconds = total_words * 0.5
                print(
                    f"Estimated audio length for async text: {estimated_length_seconds:.2f} seconds"
                )
            url = "http://127.0.0.1:5000/process-audio/"
            resp = requests.get(url)
            asyncio.create_task(handle_async_text())
        return tokenize.utils.replace_words(
            text=text, replacements={"livekit": r"<<l|aɪ|v|k|ɪ|t|>>"}
        )

    # also for this example, we also intensify the keyword "LiveKit" to make it more likely to be
    # recognized with the STT
    deepgram_stt = deepgram.STT(keywords=[("LiveKit", 3.5)])

    agent = VoicePipelineAgent(
        vad=silero.VAD.load(),
        stt=deepgram_stt,
        llm=openai.LLM(),
        tts=cartesia.TTS(),
        chat_ctx=initial_ctx,
        before_tts_cb=_before_tts_cb,
    )
    agent.start(ctx.room)
    print("agent should speak now")
    await agent.emit
    await agent.say("Hey, LiveKit is awesome!", allow_interruptions=True)
    await agent.say("taattatta", allow_interruptions=True)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
