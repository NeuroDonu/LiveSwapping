# -*- coding: utf-8 -*-
"""Основные модули обработки LiveSwapping."""

from .realtime import main as realtime_main
from .video import main as video_main
from .Image import main as image_main

__all__ = ["realtime_main", "video_main", "image_main"] 