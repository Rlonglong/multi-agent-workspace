from sqlalchemy import Column, String, Text, DateTime, JSON, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
from app.core.db import Base
import uuid

def generate_uuid():
    return str(uuid.uuid4())

class Workspace(Base):
    __tablename__ = "workspaces"
    id = Column(String, primary_key=True, index=True, default=generate_uuid)
    title = Column(String, default="New Project")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Store the dynamic swarm agent roles config for this workspace
    # e.g., [{"role": "PM", "model": "gpt-4o", "prompt": "..."}, ...]
    agent_configs = Column(JSON, default=list)
    
    # Store the central editable Guideline
    guideline_content = Column(Text, default="")
    
    messages = relationship("Message", back_populates="workspace", cascade="all, delete-orphan", order_by="Message.created_at")
    artifacts = relationship("Artifact", back_populates="workspace", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "messages"
    id = Column(String, primary_key=True, index=True, default=generate_uuid)
    workspace_id = Column(String, ForeignKey("workspaces.id"), index=True)
    
    # "user", "assistant", "system", "tool"
    role = Column(String, nullable=False)
    
    # Specific agent name "CTO" or "Frontend"
    name = Column(String, nullable=True)
    
    content = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)
    
    workspace = relationship("Workspace", back_populates="messages")


class Artifact(Base):
    __tablename__ = "artifacts"
    id = Column(String, primary_key=True, index=True, default=generate_uuid)
    workspace_id = Column(String, ForeignKey("workspaces.id"), index=True)
    
    filename = Column(String, nullable=False)
    content = Column(Text, default="")
    language = Column(String, default="text")
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    workspace = relationship("Workspace", back_populates="artifacts")
