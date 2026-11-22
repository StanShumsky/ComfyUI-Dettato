"""ComfyUI nodes for KobiMusic's Dettato multi-instrument audio-to-MIDI transcription.

The whole pack registers through the V3 entrypoint below. ComfyUI's loader registers a custom
node module as either V1 (NODE_CLASS_MAPPINGS) or V3 (comfy_entrypoint) - it checks V1 first
and returns, so a module with both would silently lose its V3 nodes; this pack therefore has no
NODE_CLASS_MAPPINGS of its own and registers everything here: the twelve stage/tool nodes and the
native-V3 DettatoTranscribe (autogrowing audio slots), all io.ComfyNode classes in nodes.py.
"""
from comfy_api.latest import ComfyExtension

from . import nodes

__all__ = ["comfy_entrypoint"]


class DettatoPack(ComfyExtension):
    async def get_node_list(self):
        return [
            nodes.DettatoLoader,
            nodes.DettatoEvidence,
            nodes.DettatoGrouper,
            nodes.DettatoDrums,
            nodes.DettatoTranscribe,
            nodes.DettatoDedupeMIDI,
            nodes.DettatoQuantizeMIDI,
            nodes.DettatoPianoRoll,
            nodes.DettatoSaveMIDI,
            nodes.DettatoInfo,
            nodes.DettatoMIDIToText,
            nodes.DettatoToYuE2ABC,
            nodes.DettatoSaveABC,
        ]


async def comfy_entrypoint() -> DettatoPack:
    return DettatoPack()
