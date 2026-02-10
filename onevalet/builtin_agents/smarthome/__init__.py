"""
Smart Home agents for OneValet

Provides agents for controlling Philips Hue lights and Sonos speakers.
"""

from .light import LightControlAgent

__all__ = ["LightControlAgent"]

try:
    from .speaker import SpeakerControlAgent
    __all__.append("SpeakerControlAgent")
except ImportError:
    pass
