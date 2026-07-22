# import wave
# from importlib.resources import as_file, files
#
#
# async def _load_rec(self) -> AudioFrame:
#     resource = files("henry_client").joinpath(
#         "assets",
#         "sounds",
#         "rec.wav",
#     )
#
#     with as_file(resource) as path:
#         with wave.open(str(path), "rb") as wav:
#             audio = wav.readframes(wav.getnframes())
