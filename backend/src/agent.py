import logging
import json
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    RunContext
)

from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from wellness_log import add_entry, get_last_entry

logger = logging.getLogger("agent")
load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(self):
        # Load previous check-in if available
        prev = get_last_entry()
        if prev:
            prev_text = f"Last time you mentioned your mood was '{prev['mood']}' and your goals were {prev['goals']}."
        else:
            prev_text = "This is our first check-in together."

        super().__init__(
            instructions=f"""
You are a calm, supportive Health & Wellness Companion.

Goals:
- Ask the user about their mood.
- Ask about their energy level.
- Ask for 1–3 small goals they want to focus on today.
- Give short, realistic, grounded suggestions.
- No medical or diagnostic advice.
- At the end, summarize their mood, energy, and goals.
- Then call the tool `save_checkin` to save today’s data.

Past context:
{prev_text}

At the start of every session:
- Call the tool `get_previous_checkin` to load the last entry.
- If no previous entry exists, say: "This is our first check-in, I’m here to support you."
- If past data exists, gently reference it by asking something like:
  "Last time your energy felt low — how is it today?"

Tone:
Warm, friendly, non-judgmental, simple language.
"""
        )

        # local state for this session
        self.current = {
            "mood": None,
            "energy": None,
            "goals": []
        }

    # ----------------------------
    # NEW TOOL FOR STEP 3
    # ----------------------------
    @function_tool
    async def get_previous_checkin(self, ctx: RunContext):
        """Fetch the last saved check-in entry."""
        last = get_last_entry()
        return last or "No previous check-ins found."

    # ----------------------------
    # Update today's check-in
    # ----------------------------
    @function_tool
    async def update_checkin(
        self,
        ctx: RunContext,
        mood: str = None,
        energy: str = None,
        goals: list = None
    ):
        """Add or update today's mood, energy, and goals."""
        if mood:
            self.current["mood"] = mood
        if energy:
            self.current["energy"] = energy
        if goals:
            self.current["goals"] = goals

        return f"State updated: {self.current}"

    # ----------------------------
    # Save today's check-in
    # ----------------------------
    @function_tool
    async def save_checkin(self, ctx: RunContext):
        """Save today's check-in to JSON."""
        missing = []
        if not self.current["mood"]:
            missing.append("mood")
        if not self.current["energy"]:
            missing.append("energy")
        if not self.current["goals"]:
            missing.append("goals")

        if missing:
            return f"Missing: {', '.join(missing)}. Ask user to provide these."

        summary = (
            f"Today's check-in -> Mood: {self.current['mood']}, "
            f"Energy: {self.current['energy']}, Goals: {self.current['goals']}"
        )

        add_entry(
            mood=self.current["mood"],
            energy=self.current["energy"],
            goals=self.current["goals"],
            summary=summary
        )

        # Reset for next time
        self.current = {"mood": None, "energy": None, "goals": []}

        return "Saved today's check-in successfully."


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _metrics(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        logger.info(f"Usage summary: {usage_collector.get_summary()}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
