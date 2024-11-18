from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Dict, Literal, Annotated, Optional, List
from agent_state import AgentState
from prompt import PROMPT
from tools import TOOLS, logger
import inspect

from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    WorkerType,
    cli,
    llm,
)
from livekit.agents.multimodal import MultimodalAgent
from livekit.plugins import openai
from livekit.plugins.openai.realtime import api_proto


# logger = logging.getLogger("my-worker")
# logger.setLevel(logging.INFO)


#***********************************************

from database import UsageLog, ConversationLog, init_db  # Importing the UsageLog model and init_db function
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# Database configuration
DATABASE_URL = "sqlite+aiosqlite:///./livekit_db.sqlite"

# Create the async engine
engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    future=True,  # Enable future SQLAlchemy features
)

# Create the async session maker
async_session = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)

async def log_usage( usage_details: Dict[str, Any]):
    try:
        input_text_tokens = usage_details["input_token_details"]["text_tokens"]
        input_audio_tokens = usage_details["input_token_details"]["audio_tokens"]
        output_text_tokens = usage_details["output_token_details"]["text_tokens"]
        output_audio_tokens = usage_details["output_token_details"]["audio_tokens"]

        async with async_session() as session:
            async with session.begin():
                usage_log = UsageLog(
                    input_text_tokens=input_text_tokens,
                    input_audio_tokens=input_audio_tokens,
                    output_text_tokens=output_text_tokens,
                    output_audio_tokens=output_audio_tokens,
                    timestamp=datetime.utcnow()
                )
                session.add(usage_log)
        logger.debug(f"Successfully logged usage for job_id: ")

    except Exception as e:
        logger.error(f"Failed to log usage: {e}")


async def log_conversation(conversation_history, input_text, input_audio, output_text, output_audio):
    try:
        async with async_session() as session:
            async with session.begin():
                conversation_log = ConversationLog(
                    bot_name='SK-Bot',
                    conversation=conversation_history,
                    session_input_text_tokens = input_text,
                    session_input_audio_tokens = input_audio,
                    session_output_text_tokens = output_text,
                    session_output_audio_tokens = output_audio,
                    timestamp=datetime.utcnow()
                )
                session.add(conversation_log)
        logger.debug("Successfully logged conversation")

    except Exception as e:
        logger.error(f"Failed to log conversation: {e}")

#*************************************************************




@dataclass
class SessionConfig:
    openai_api_key: str
    instructions: str
    voice: openai.realtime.api_proto.Voice
    temperature: float
    max_response_output_tokens: str | int
    modalities: list[openai.realtime.api_proto.Modality]
    turn_detection: openai.realtime.ServerVadOptions

    def __post_init__(self):
        if self.modalities is None:
            self.modalities = self._modalities_from_string("text_and_audio")

    def to_dict(self):
        return {k: v for k, v in asdict(self).items() if k != "openai_api_key"}

    @staticmethod
    def _modalities_from_string(modalities: str) -> list[str]:
        modalities_map = {
            "text_and_audio": ["text", "audio"],
            "text_only": ["text"],
        }
        return modalities_map.get(modalities, ["text", "audio"])

    def __eq__(self, other: SessionConfig) -> bool:
        return self.to_dict() == other.to_dict()


def parse_session_config(data: Dict[str, Any]) -> SessionConfig:
    turn_detection = None

    if data.get("turn_detection"):
        turn_detection_json = json.loads(data.get("turn_detection"))
        turn_detection = openai.realtime.ServerVadOptions(
            threshold=turn_detection_json.get("threshold", 0.5),
            prefix_padding_ms=turn_detection_json.get("prefix_padding_ms", 200),
            silence_duration_ms=turn_detection_json.get("silence_duration_ms", 300),
        )
    else:
        turn_detection = openai.realtime.DEFAULT_SERVER_VAD_OPTIONS

    config = SessionConfig(
        openai_api_key=data.get("openai_api_key", ""),
        instructions=data.get("instructions", ""),
        voice=data.get("voice", "alloy"),
        temperature=float(data.get("temperature", 0.8)),
        max_response_output_tokens=data.get("max_output_tokens")
        if data.get("max_output_tokens") == "inf"
        else int(data.get("max_output_tokens") or 2048),
        modalities=SessionConfig._modalities_from_string(
            data.get("modalities", "text_and_audio")
        ),
        turn_detection=turn_detection,
    )
    return config


async def entrypoint(ctx: JobContext):
    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    participant = await ctx.wait_for_participant()

    run_multimodal_agent(ctx, participant)

    logger.info("agent started")


"""
    ********************************
    Class to add tools
    ********************************

"""
class AssistantFunctions(llm.FunctionContext):
    
    # Initialize the agent state
    # agent_state = AgentState(
    #     destination_point=[],
    #     start_point=[],
    #     quantity=0,
    #     type="",
    #     contents="",
    #     customer_type="",
    # )

    def __init__(self):
        super().__init__()

        self.agent_state = AgentState(
            destination_point=None,
            start_point=None,
            quantity=None,
            type=None,
            contents=None,
            customer_name=None,
            weight=None,
            dimensions=None,
            support_services=None,
            planned_date=None,
        )
        self.register_dynamic_functions()


    def register_dynamic_functions(self):
        for func in TOOLS:
            func_name = func.__name__

            # Get the signature of func
            sig = inspect.signature(func)
            # Remove 'self' from parameters
            parameters = list(sig.parameters.values())
            parameters = [p for p in parameters if p.name != 'self']
            new_sig = sig.replace(parameters=parameters)

            # Define a function to create the wrapper
            def create_wrapper(func=func, func_name=func_name, new_sig=new_sig):
                async def wrapper(*args, **kwargs):
                    return await func(self, *args, **kwargs)
                wrapper.__name__ = func_name
                wrapper.__signature__ = new_sig
                if hasattr(func, '__annotations__'):
                    func_annotations = func.__annotations__
                    wrapper_annotations = {k: v for k, v in func_annotations.items() if k != 'self'}
                    wrapper.__annotations__ = wrapper_annotations
                return wrapper

            # Create the wrapper
            wrapper = create_wrapper()

            # Apply the decorator
            decorated_func = llm.ai_callable()(wrapper)

            # Add the method to the instance
            setattr(self, func_name, decorated_func)

            # Register the function with FunctionContext
            self._register_ai_function(decorated_func)


    # @llm.ai_callable()
    # async def update_start_and_destination(
    #     self,
    #     start_point: Annotated[str, llm.TypeInfo(description="Point of Load or starting point")],
    #     destination_point: Annotated[str, llm.TypeInfo(description="Point of Discharge or destination point")],
    # ):
    #     """Updates the start and destination points in the agent state."""
    #     self.agent_state['start_point'] = start_point
    #     self.agent_state['destination_point'] = destination_point
    #     logger.info(f"Updated start and destination points: {self.agent_state}")
    #     return json.dumps({"status": "Start and destination points updated"})


    # @llm.ai_callable()
    # async def update_quantity_and_type(
    #     self,
    #     quantity: Annotated[int, llm.TypeInfo(description="Quantity of containers or packages")],
    #     type: Annotated[str, llm.TypeInfo(description="Container type (e.g., '20-foot container') or dimensions for LCL")],
    # ):
    #     """Updates the quantity and type in the agent state."""
    #     self.agent_state['quantity'] = quantity
    #     self.agent_state['type'] = type
    #     logger.info(f"Updated quantity and type: {self.agent_state}")
    #     return json.dumps({"status": "Quantity and type updated"})


    # @llm.ai_callable()
    # async def update_weight_and_dimensions(
    #     self,
    #     weight: Annotated[float, llm.TypeInfo(description="Weight in kilograms")],
    #     dimensions: Annotated[str, llm.TypeInfo(description="Dimensions in Length x Breadth x Height format")],
    # ):
    #     """Updates the weight and dimensions in the agent state."""
    #     self.agent_state['weight'] = weight
    #     self.agent_state['dimensions'] = dimensions
    #     logger.info(f"Updated weight and dimensions: {self.agent_state}")
    #     return json.dumps({"status": "Weight and dimensions updated"})


    # @llm.ai_callable()
    # async def update_contents(
    #     self,
    #     contents: Annotated[str, llm.TypeInfo(description="Contents of the cargo")],
    # ):
    #     """Updates the contents in the agent state."""
    #     self.agent_state['contents'] = contents
    #     logger.info(f"Updated contents: {self.agent_state}")
    #     return json.dumps({"status": "Contents updated"})


    # @llm.ai_callable()
    # async def update_customer_name(
    #     self,
    #     customer_name: Annotated[str, llm.TypeInfo(description="Name of the customer")],
    # ):
    #     """Updates the customer name in the agent state."""
    #     self.agent_state['customer_name'] = customer_name
    #     logger.info(f"Updated customer name: {self.agent_state}")
    #     return json.dumps({"status": "Customer name updated"})
    

    # @llm.ai_callable()
    # async def update_support_services(
    #     self,
    #     support_services: Annotated[str, llm.TypeInfo(description="List of support services required, e.g., ['Warehousing', 'Transportation']")],
    # ):
    #     """Updates the support services in the agent state."""
    #     self.agent_state['support_services'] = support_services
    #     logger.info(f"Updated support services: {self.agent_state}")
    #     return json.dumps({"status": "Support services updated"})


    # @llm.ai_callable()
    # async def update_planned_date(
    #     self,
    #     planned_date: Annotated[str, llm.TypeInfo(description="Planned date of shipment")],
    # ):
    #     """Updates the planned date in the agent state."""
    #     self.agent_state['planned_date'] = planned_date
    #     logger.info(f"Updated planned date: {self.agent_state}")
    #     return json.dumps({"status": "Planned date updated"})




    # @llm.ai_callable()
    # async def state_updation(
    #     self,
    #     destination_point: Optional[Annotated[str, llm.TypeInfo(description="Point of discharge or the destination point of the container")]] = None,
    #     start_point: Optional[Annotated[str, llm.TypeInfo(description="Point of load or the starting point of the container")]] = None,
    #     quantity: Optional[Annotated[int, llm.TypeInfo(description="Size of container in feets")]] = None,
    #     container_type: Optional[Annotated[str, llm.TypeInfo(description="Type of the container")]] = None,
    #     contents: Optional[Annotated[str, llm.TypeInfo(description="Contents type that container will carry")]] = None,
    #     customer_name: Optional[Annotated[str, llm.TypeInfo(description="Name of customer")]] = None,
    # ):
    #     """Updates the agent state based on provided arguments, allowing dynamic updates without exceeding defined state attributes."""
        
    #     # Loop through each provided argument and update the agent state
    #     for key, value in {
    #         "destination_point": destination_point, 
    #         "start_point": start_point, 
    #         "quantity": quantity, 
    #         "type": container_type, 
    #         "contents": contents, 
    #         "customer_type": customer_name
    #     }.items():
    #         if value is not None:
    #             self.agent_state[key] = value
        
    #     # Return success message and updated state for verification
    #     logger.info(f"\nUpdated state: {self.agent_state}\n\n")
    #     print(f"\nUpdated state: {self.agent_state}\n\n")

    #     return json.dumps({"status": "State updated successfully", "updated_state": self.agent_state})



        


    # You can include methods to update the agent state



def run_multimodal_agent(ctx: JobContext, participant: rtc.Participant):
    metadata = json.loads(participant.metadata)
    config = parse_session_config(metadata)
    logger.info(f"starting omni assistant with config: {config.to_dict()}")


    # Instantiate your function context with the agent state
    fnc_ctx = AssistantFunctions()

    conversation_history = []
    session_input_text_tokens = 0
    session_input_audio_tokens = 0
    session_output_text_tokens = 0
    session_output_audio_tokens = 0
    



    model = openai.realtime.RealtimeModel(
        api_key=config.openai_api_key,
        instructions=PROMPT,
        # instructions=config.instructions,
        voice=config.voice,
        temperature=config.temperature,
        max_response_output_tokens=config.max_response_output_tokens,
        modalities=config.modalities,
        turn_detection=config.turn_detection,
        # Pass the function context
    )
    assistant = MultimodalAgent(model=model, fnc_ctx=fnc_ctx)
    assistant.start(ctx.room)
    session = model.sessions[0]

    if config.modalities == ["text", "audio"]:
        session.conversation.item.create(
            llm.ChatMessage(
                role="user",
                content="Please begin the interaction with the user in a manner consistent with your instructions.",
            )
        )
        session.response.create()

    @ctx.room.local_participant.register_rpc_method("pg.updateConfig")
    async def update_config(
        data: rtc.rpc.RpcInvocationData,
    ):
        if data.caller_identity != participant.identity:
            return

        new_config = parse_session_config(json.loads(data.payload))
        if config != new_config:
            logger.info(
                f"config changed: {new_config.to_dict()}, participant: {participant.identity}"
            )
            session = model.sessions[0]
            session.session_update(
                instructions=new_config.instructions,
                voice=new_config.voice,
                temperature=new_config.temperature,
                max_response_output_tokens=new_config.max_response_output_tokens,
                turn_detection=new_config.turn_detection,
                modalities=new_config.modalities,
            )
            return json.dumps({"changed": True})
        else:
            return json.dumps({"changed": False})

    @session.on("response_done")
    def on_response_done(response: openai.realtime.RealtimeResponse):
        nonlocal session_input_text_tokens, session_input_audio_tokens
        nonlocal session_output_text_tokens, session_output_audio_tokens

        variant: Literal["warning", "destructive"]
        description: str | None = None
        title: str
        if response.status == "incomplete":
            if response.status_details and response.status_details["reason"]:
                reason = response.status_details["reason"]
                if reason == "max_output_tokens":
                    variant = "warning"
                    title = "Max output tokens reached"
                    description = "Response may be incomplete"
                elif reason == "content_filter":
                    variant = "warning"
                    title = "Content filter applied"
                    description = "Response may be incomplete"
                else:
                    variant = "warning"
                    title = "Response incomplete"
            else:
                variant = "warning"
                title = "Response incomplete"
        elif response.status == "failed":
            if response.status_details and response.status_details["error"]:
                error_code = response.status_details["error"]["code"]
                if error_code == "server_error":
                    variant = "destructive"
                    title = "Server error"
                elif error_code == "rate_limit_exceeded":
                    variant = "destructive"
                    title = "Rate limit exceeded"
                else:
                    variant = "destructive"
                    title = "Response failed"
            else:
                variant = "destructive"
                title = "Response failed"
        
        elif response.status == "completed":
            if response.usage:
                # Instead of sending to frontend, store in the database
                # job_id = ctx.job_id  # Assuming ctx has job_id
                # pid = os.getpid()  # Get current process ID
                usage_details = response.usage  # Assuming response.usage is a dict

                
                session_input_text_tokens += usage_details["input_token_details"]["text_tokens"]
                session_input_audio_tokens += usage_details["input_token_details"]["audio_tokens"]
                session_output_text_tokens += usage_details["output_token_details"]["text_tokens"]
                session_output_audio_tokens += usage_details["output_token_details"]["audio_tokens"]
    

                # Schedule the database logging
                asyncio.create_task(log_usage(usage_details))

                # Optionally, you can log it locally as well
                logger.info(f"Usage logged to database: {usage_details}")

                title = "Response completed"
                variant = "success"
        #Added this elif for usage tracking
        # elif response.status == "completed":
        #     print(f"Response is: {response}")
        #     if response.usage:
        #         title = response.usage
        #         variant = "success"
        #         description = "Token usage"

        else:
            return

        asyncio.create_task(show_toast(title, description, variant))

    async def send_transcription(
        ctx: JobContext,
        participant: rtc.Participant,
        track_sid: str,
        segment_id: str,
        text: str,
        is_final: bool = True,
    ):
        transcription = rtc.Transcription(
            participant_identity=participant.identity,
            track_sid=track_sid,
            segments=[
                rtc.TranscriptionSegment(
                    id=segment_id,
                    text=text,
                    start_time=0,
                    end_time=0,
                    language="en",
                    final=is_final,
                )
            ],
        )
        await ctx.room.local_participant.publish_transcription(transcription)

    async def show_toast(
        title: str,
        description: str | None,
        variant: Literal["default", "success", "warning", "destructive"],
    ):
        await ctx.room.local_participant.perform_rpc(
            destination_identity=participant.identity,
            method="pg.toast",
            payload=json.dumps(
                {"title": title, "description": description, "variant": variant}
            ),
        )

    last_transcript_id = None

    # send three dots when the user starts talking. will be cleared later when a real transcription is sent.
    @session.on("input_speech_started")
    def on_input_speech_started():
        nonlocal last_transcript_id
        remote_participant = next(iter(ctx.room.remote_participants.values()), None)
        if not remote_participant:
            return

        track_sid = next(
            (
                track.sid
                for track in remote_participant.track_publications.values()
                if track.source == rtc.TrackSource.SOURCE_MICROPHONE
            ),
            None,
        )
        if last_transcript_id:
            asyncio.create_task(
                send_transcription(
                    ctx, remote_participant, track_sid, last_transcript_id, ""
                )
            )

        new_id = str(uuid.uuid4())
        last_transcript_id = new_id
        asyncio.create_task(
            send_transcription(
                ctx, remote_participant, track_sid, new_id, "…", is_final=False
            )
        )

    @session.on("input_speech_transcription_completed")
    def on_input_speech_transcription_completed(
        event: openai.realtime.InputTranscriptionCompleted,
    ):
        logger.info(f"Input transcript completed: {event.transcript}")
        conversation_history.append({
                'role': 'user',
                'content': event.transcript
            })

        nonlocal last_transcript_id
        if last_transcript_id:
            remote_participant = next(iter(ctx.room.remote_participants.values()), None)
            if not remote_participant:
                return

            track_sid = next(
                (
                    track.sid
                    for track in remote_participant.track_publications.values()
                    if track.source == rtc.TrackSource.SOURCE_MICROPHONE
                ),
                None,
            )
            asyncio.create_task(
                send_transcription(
                    ctx, remote_participant, track_sid, last_transcript_id, ""
                )
            )
            last_transcript_id = None

    @session.on("input_speech_transcription_failed")
    def on_input_speech_transcription_failed(
        event: openai.realtime.InputTranscriptionFailed,
    ):
        nonlocal last_transcript_id
        if last_transcript_id:
            remote_participant = next(iter(ctx.room.remote_participants.values()), None)
            if not remote_participant:
                return

            track_sid = next(
                (
                    track.sid
                    for track in remote_participant.track_publications.values()
                    if track.source == rtc.TrackSource.SOURCE_MICROPHONE
                ),
                None,
            )

            error_message = "⚠️ Transcription failed"
            asyncio.create_task(
                send_transcription(
                    ctx,
                    remote_participant,
                    track_sid,
                    last_transcript_id,
                    error_message,
                )
            )
            last_transcript_id = None


    @session.on("response_audio_transcript_done")
    def on_response_audio_transcript_done(
        content: str
    ):
        # Process the conversation item here
        logger.info(f"Output transcript completed: {content}")
        conversation_history.append({
                'role': 'assistant',
                'content': content
        })

        # asyncio.create_task(log_conversation(conversation_history))


    @session.on("conversation_item_created")
    def on_conversation_item_created(
        item: api_proto.Resource.Item
    ):
        # Process the conversation item here
        logger.info(f"Conversation item created: {item}")

    
    async def on_shutdown():
        logger.info("Job is shutting down, logging conversation.")

        # Since appending to the list is thread-safe, and we're only reading from it here,
        # we can proceed without a lock.
        await log_conversation(conversation_history, session_input_text_tokens, session_input_audio_tokens, session_output_text_tokens, session_output_audio_tokens)
        logger.info("Conversation logged successfully.")

    ctx.add_shutdown_callback(on_shutdown)
        




if __name__ == "__main__":
    asyncio.run(init_db(engine))
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, worker_type=WorkerType.ROOM))
