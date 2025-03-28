from __future__ import annotations

import json
from typing import AsyncIterable
from dotenv import load_dotenv
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    llm,
    tokenize,
    metrics,
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import deepgram, openai, silero
import requests

load_dotenv()

input_text = "In a land far beyond the reach of time, nestled between towering mountains and vast oceans, lay the Enchanted Forest. This forest was unlike any other, filled with trees that whispered secrets, flowers that glowed in the dark, and streams that sang lullabies. The forest was home to many magical creatures, but none were as curious and adventurous as a young elf named Elara.Elara had always been fascinated by the stories of the ancient times, tales of great heroes, lost treasures, and powerful spells. She spent her days exploring the forest, hoping to uncover its many mysteries. One sunny morning, as she wandered deeper into the woods than ever before, she stumbled upon a hidden path covered in ivy and moss. Intrigued, she decided to follow it. The path led her to a clearing where a magnificent tree stood. Its trunk was wide and gnarled, and its branches stretched high into the sky, covered in shimmering leaves. At the base of the tree, Elara noticed a small, ornate door. Her heart raced with excitement as she reached out and gently pushed it open. Inside, she found a spiral staircase leading down into the earth. With a deep breath, she descended the stairs, her footsteps echoing softly. At the bottom, she entered a vast underground chamber filled with shelves upon shelves of ancient books and scrolls. In the center of the room stood a pedestal with a single, glowing book. Elara approached the book and carefully opened it. The pages were filled with intricate illustrations and strange symbols. As she read, she realized it was a spellbook, containing powerful magic that had been lost for centuries. One spell, in particular, caught her eye – a spell to summon the Guardian of the Forest, a mythical creature said to possess immense wisdom and strength. Determined to learn more, Elara spent days studying the spellbook, practicing the incantations and gathering the necessary ingredients. Finally, she was ready. Under the light of a full moon, she returned to the clearing and began the ritual. She chanted the ancient words, and as she did, the ground trembled and the air crackled with energy. Suddenly, a blinding light filled the clearing, and when it faded, a majestic creature stood before her. The Guardian of the Forest was a magnificent stag with antlers that glowed like the stars and eyes that shimmered with ancient knowledge. Elara bowed respectfully, and the Guardian spoke in a deep, resonant voice. 'Young elf, you have summoned me. What is it that you seek?' Elara explained her desire to protect the forest and learn its secrets. The Guardian nodded and agreed to teach her, but only if she proved herself worthy. He set her three tasks, each more challenging than the last. For the first task, Elara had to find the Heart of the Forest, a rare and precious gem hidden deep within the woods. She faced many dangers along the way, from treacherous terrain to cunning creatures, but her determination and resourcefulness saw her through. She found the gem and returned it to the Guardian. The second task required her to heal a wounded dragon that had been terrorizing a nearby village. Elara used her knowledge of herbs and potions to create a healing elixir. She approached the dragon with caution and compassion, earning its trust and administering the elixir. The dragon was healed and flew away, no longer a threat to the villagers. For the final task, Elara had to confront her greatest fear – the Shadow of Doubt, a dark entity that fed on insecurity and fear. She journeyed to the Shadow's lair and faced it head-on. Drawing on her inner strength and the lessons she had learned, she banished the Shadow, freeing herself from its grip. The Guardian was impressed by Elara's courage and wisdom. He bestowed upon her the title of Protector of the Forest and granted her the ability to communicate with all the creatures of the woods. Elara continued to explore and protect the Enchanted Forest, her heart filled with gratitude and wonder. And so, the legend of Elara, the brave elf who became the Protector of the Forest, was passed down through generations, inspiring countless others to seek adventure and protect the magic of the world around them."


async def entrypoint(ctx: JobContext):
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            "You are a voice assistant created by LiveKit. Your interface with users will be voice. "
            "You should use short and concise responses, and avoiding usage of unpronouncable punctuation."
        ),
    )
    await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL)

    async def _before_tts_cb(agent: VoicePipelineAgent, text: str | AsyncIterable[str]):
        url = "http://127.0.0.1:5000/process-text"
        headers = {"Accept": "*/*", "Content-Type": "application/json"}
        resp = requests.post(url, headers=headers, json={"text": input_text})
        if resp.status_code != 200:
            print(f"Failed to process text: {resp.text}")
        new_input_text = json.loads(resp.text)["text"]
        return tokenize.utils.replace_words(
            text=new_input_text, replacements={"livekit": r"<<l|aɪ|v|k|ɪ|t|>>"}
        )

    # also for this example, we also intensify the keyword "LiveKit" to make it more likely to be
    # recognized with the STT
    deepgram_stt = deepgram.STT(keywords=[("LiveKit", 3.5)])

    agent = VoicePipelineAgent(
        vad=silero.VAD.load(),
        stt=deepgram_stt,
        llm=openai.LLM(),
        tts=openai.TTS(),
        chat_ctx=initial_ctx,
        before_tts_cb=_before_tts_cb,
    )
    agent.start(ctx.room)
    await agent.say(
        input_text,
        allow_interruptions=True,
    )
    usage_collector = metrics.UsageCollector()

    @agent.on("metrics_collected")
    def _on_metrics_collected(mtrcs: metrics.AgentMetrics):
        metrics.log_metrics(mtrcs)
        usage_collector.collect(mtrcs)

    async def log_usage():
        summary = usage_collector.get_summary()
        print(f"Usage: ${summary}")

    ctx.add_shutdown_callback(log_usage)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
