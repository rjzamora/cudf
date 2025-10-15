# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""ChannelPair abstraction for metadata + data channels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias

from rapidsmpf.streaming.core.channel import Channel, Message
from rapidsmpf.streaming.core.pyobject import PyObjectPayload

if TYPE_CHECKING:
    from rapidsmpf.streaming.core.context import Context
    from rapidsmpf.streaming.cudf.table_chunk import TableChunk


# Type alias for metadata payloads
Metadata: TypeAlias = PyObjectPayload


@dataclass
class ChannelPair:
    """
    A pair of channels for metadata and table data.

    This abstraction ensures that metadata and data are kept separate,
    avoiding ordering issues and making the code more type-safe.

    Attributes
    ----------
    metadata : Channel[Metadata]
        Channel for metadata (PyObjectPayload).
    data : Channel[TableChunk]
        Channel for table data chunks.
    """

    metadata: Channel[Metadata]
    data: Channel[TableChunk]

    @classmethod
    def create(cls) -> ChannelPair:
        """Create a new ChannelPair with fresh channels."""
        return cls(metadata=Channel(), data=Channel())

    async def drain_both(self, ctx: Context) -> None:
        """Drain both the metadata and data channels."""
        await self.metadata.drain(ctx)
        await self.data.drain(ctx)

    async def shutdown_both(self, ctx: Context) -> None:
        """Shutdown both the metadata and data channels."""
        await self.metadata.shutdown(ctx)
        await self.data.shutdown(ctx)

    async def send_metadata(
        self, ctx: Context, metadata: dict[str, Any] | None
    ) -> None:
        """
        Send metadata if present, then drain metadata channel.

        Parameters
        ----------
        ctx : Context
            The streaming context.
        metadata : dict | None
            The metadata to send. If None, just drain.
        """
        if metadata is not None:
            payload = PyObjectPayload.from_object(
                sequence_number=0,
                obj=metadata,
            )
            await self.metadata.send(ctx, Message(payload))
        await self.metadata.drain(ctx)

    async def recv_metadata(self, ctx: Context) -> dict[str, Any] | None:
        """
        Receive metadata from the metadata channel.

        Parameters
        ----------
        ctx : Context
            The streaming context.

        Returns
        -------
        dict | None
            The metadata dictionary, or None if channel is drained.
        """
        msg = await self.metadata.recv(ctx)
        if msg is None:
            return None
        payload = PyObjectPayload.from_message(msg)
        return payload.get_object()


def create_channel_pair() -> ChannelPair:
    """
    Helper function to create a new ChannelPair.

    Returns
    -------
    ChannelPair
        A new ChannelPair with fresh metadata and data channels.
    """
    return ChannelPair.create()
