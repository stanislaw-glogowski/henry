import asyncio

type SpeechCommand = None

type SpeechCommands = asyncio.Queue[SpeechCommand]
