"""API endpoints for agent messaging."""

from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.agents.core import (
    Message,
    MessagePriority,
    MessageType,
    get_message_router,
)

router = APIRouter()


class SendMessageRequest(BaseModel):
    """Request to send a message."""

    sender_id: str
    recipient_id: str | None = None
    subject: str
    content: Any
    msg_type: str = "direct"
    priority: int = 1
    groups: list[str] | None = None


class BroadcastRequest(BaseModel):
    """Request to broadcast a message."""

    sender_id: str
    subject: str
    content: Any
    priority: int = 1


class GroupMessageRequest(BaseModel):
    """Request to send to a group."""

    sender_id: str
    group: str
    subject: str
    content: Any
    priority: int = 1


class KnowledgeShareRequest(BaseModel):
    """Request to share knowledge."""

    sender_id: str
    recipient_id: str | None = None
    knowledge_type: str
    content: Any
    embedding: list[float] | None = None


class CollaborationRequest(BaseModel):
    """Request for collaboration."""

    sender_id: str
    recipient_id: str
    task: str
    details: Any


# -----------------------------------------------------------------------------
# Message Endpoints
# -----------------------------------------------------------------------------


@router.post("/send")
async def send_message(request: SendMessageRequest) -> dict[str, Any]:
    """Send a message to another agent."""
    router_instance = get_message_router()
    if not router_instance:
        raise HTTPException(status_code=503, detail="Message router not initialized")

    try:
        msg_type = MessageType(request.msg_type)
    except ValueError:
        msg_type = MessageType.DIRECT

    try:
        priority = MessagePriority(request.priority)
    except ValueError:
        priority = MessagePriority.NORMAL

    msg = Message(
        type=msg_type,
        priority=priority,
        sender_id=UUID(request.sender_id),
        recipient_id=UUID(request.recipient_id) if request.recipient_id else None,
        subject=request.subject,
        content=request.content,
    )

    if request.groups:
        msg.metadata["groups"] = request.groups

    success = router_instance.send(msg)

    return {
        "success": success,
        "message": msg.to_dict(),
    }


@router.post("/broadcast")
async def broadcast_message(request: BroadcastRequest) -> dict[str, Any]:
    """Broadcast a message to all agents."""
    router_instance = get_message_router()
    if not router_instance:
        raise HTTPException(status_code=503, detail="Message router not initialized")

    msg = router_instance.broadcast(
        sender_id=UUID(request.sender_id),
        subject=request.subject,
        content=request.content,
        priority=MessagePriority(request.priority),
    )

    return {
        "success": True,
        "message": msg.to_dict(),
        "recipients": len(router_instance.agents) - 1,
    }


@router.post("/group/{group}")
async def send_to_group(group: str, request: GroupMessageRequest) -> dict[str, Any]:
    """Send a message to all agents in a group."""
    router_instance = get_message_router()
    if not router_instance:
        raise HTTPException(status_code=503, detail="Message router not initialized")

    msg = router_instance.send_to_group(
        sender_id=UUID(request.sender_id),
        group=group,
        subject=request.subject,
        content=request.content,
        priority=MessagePriority(request.priority),
    )

    members = router_instance.get_group_members(group)

    return {
        "success": True,
        "message": msg.to_dict(),
        "group": group,
        "recipients": len(members),
    }


@router.get("/inbox/{agent_id}")
async def get_inbox(
    agent_id: UUID,
    unread_only: bool = True,
    limit: int = 50,
) -> dict[str, Any]:
    """Get messages for an agent."""
    router_instance = get_message_router()
    if not router_instance:
        raise HTTPException(status_code=503, detail="Message router not initialized")

    messages = router_instance.get_messages(
        agent_id=agent_id,
        unread_only=unread_only,
        limit=limit,
    )

    return {
        "agent_id": str(agent_id),
        "count": len(messages),
        "messages": [m.to_dict() for m in messages],
    }


@router.get("/inbox/{agent_id}/by-type/{msg_type}")
async def get_messages_by_type(agent_id: UUID, msg_type: str) -> dict[str, Any]:
    """Get messages of a specific type."""
    router_instance = get_message_router()
    if not router_instance:
        raise HTTPException(status_code=503, detail="Message router not initialized")

    try:
        mtype = MessageType(msg_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid message type: {msg_type}")

    messages = router_instance.get_messages_by_type(agent_id=agent_id, msg_type=mtype)

    return {
        "agent_id": str(agent_id),
        "type": msg_type,
        "count": len(messages),
        "messages": [m.to_dict() for m in messages],
    }


@router.post("/inbox/{agent_id}/mark-read/{message_id}")
async def mark_read(agent_id: UUID, message_id: UUID) -> dict[str, Any]:
    """Mark a message as read."""
    router_instance = get_message_router()
    if not router_instance:
        raise HTTPException(status_code=503, detail="Message router not initialized")

    success = router_instance.mark_read(agent_id=agent_id, message_id=message_id)

    return {
        "success": success,
        "agent_id": str(agent_id),
        "message_id": str(message_id),
    }


@router.post("/inbox/{agent_id}/mark-all-read")
async def mark_all_read(agent_id: UUID) -> dict[str, Any]:
    """Mark all messages as read for an agent."""
    router_instance = get_message_router()
    if not router_instance:
        raise HTTPException(status_code=503, detail="Message router not initialized")

    count = router_instance.mark_all_read(agent_id=agent_id)

    return {
        "success": True,
        "agent_id": str(agent_id),
        "marked_count": count,
    }


# -----------------------------------------------------------------------------
# Knowledge Sharing
# -----------------------------------------------------------------------------


@router.post("/knowledge/share")
async def share_knowledge(request: KnowledgeShareRequest) -> dict[str, Any]:
    """Share knowledge with another agent."""
    router_instance = get_message_router()
    if not router_instance:
        raise HTTPException(status_code=503, detail="Message router not initialized")

    msg = router_instance.share_knowledge(
        sender_id=UUID(request.sender_id),
        recipient_id=UUID(request.recipient_id) if request.recipient_id else None,
        knowledge_type=request.knowledge_type,
        content=request.content,
        embedding=request.embedding,
    )

    return {
        "success": True,
        "message": msg.to_dict(),
    }


@router.post("/knowledge/query")
async def query_knowledge(
    sender_id: str,
    recipient_id: str | None = None,
    query: str = "",
) -> dict[str, Any]:
    """Query another agent for knowledge."""
    router_instance = get_message_router()
    if not router_instance:
        raise HTTPException(status_code=503, detail="Message router not initialized")

    msg = router_instance.query_knowledge(
        sender_id=UUID(sender_id),
        recipient_id=UUID(recipient_id) if recipient_id else None,
        query=query,
    )

    return {
        "success": True,
        "message": msg.to_dict(),
    }


# -----------------------------------------------------------------------------
# Collaboration
# -----------------------------------------------------------------------------


@router.post("/collaborate")
async def request_collaboration(request: CollaborationRequest) -> dict[str, Any]:
    """Request collaboration from another agent."""
    router_instance = get_message_router()
    if not router_instance:
        raise HTTPException(status_code=503, detail="Message router not initialized")

    msg = router_instance.request(
        sender_id=UUID(request.sender_id),
        recipient_id=UUID(request.recipient_id),
        request_type=request.task,
        content=request.details,
    )

    return {
        "success": True,
        "message": msg.to_dict(),
    }


# -----------------------------------------------------------------------------
# Groups & Stats
# -----------------------------------------------------------------------------


@router.get("/groups")
async def list_groups() -> dict[str, Any]:
    """List all message groups."""
    router_instance = get_message_router()
    if not router_instance:
        raise HTTPException(status_code=503, detail="Message router not initialized")

    return {
        "groups": {
            group: [str(aid) for aid in members]
            for group, members in router_instance.groups.items()
        },
    }


@router.post("/groups/{group}/add/{agent_id}")
async def add_to_group(group: str, agent_id: UUID) -> dict[str, Any]:
    """Add an agent to a group."""
    router_instance = get_message_router()
    if not router_instance:
        raise HTTPException(status_code=503, detail="Message router not initialized")

    success = router_instance.add_to_group(agent_id=agent_id, group=group)

    return {
        "success": success,
        "group": group,
        "agent_id": str(agent_id),
    }


@router.post("/groups/{group}/remove/{agent_id}")
async def remove_from_group(group: str, agent_id: UUID) -> dict[str, Any]:
    """Remove an agent from a group."""
    router_instance = get_message_router()
    if not router_instance:
        raise HTTPException(status_code=503, detail="Message router not initialized")

    success = router_instance.remove_from_group(agent_id=agent_id, group=group)

    return {
        "success": success,
        "group": group,
        "agent_id": str(agent_id),
    }


@router.get("/stats")
async def get_router_stats() -> dict[str, Any]:
    """Get message router statistics."""
    router_instance = get_message_router()
    if not router_instance:
        raise HTTPException(status_code=503, detail="Message router not initialized")

    return router_instance.get_router_stats()


@router.get("/stats/{agent_id}")
async def get_agent_stats(agent_id: UUID) -> dict[str, Any]:
    """Get message stats for an agent."""
    router_instance = get_message_router()
    if not router_instance:
        raise HTTPException(status_code=503, detail="Message router not initialized")

    stats = router_instance.get_agent_stats(agent_id)
    if stats is None:
        raise HTTPException(status_code=404, detail="Agent not found in router")

    return {
        "agent_id": str(agent_id),
        **stats,
    }
