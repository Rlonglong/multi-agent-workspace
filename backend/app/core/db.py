import os
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import declarative_base
import redis.asyncio as redis

# Default to async sqlite for rapid prototyping out-of-the-box. Handled nicely by SQLAlchemy!
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./multi_agent.db")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

Base = declarative_base()

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

async def get_redis():
    return await redis.from_url(REDIS_URL)

async def init_db():
    from app.models.session import Base
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
