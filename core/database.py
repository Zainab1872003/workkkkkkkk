# core/database.py
"""
MongoDB database connection and initialization
"""
from motor.motor_asyncio import AsyncIOMotorClient
from core.config import settings

# MongoDB client
_client = None
_db = None
_collection = None

def get_mongo_client():
    """Get MongoDB client instance"""
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(settings.MONGODB_URL)
    return _client

def get_database():
    """Get database instance"""
    global _db
    if _db is None:
        client = get_mongo_client()
        _db = client[settings.DATABASE_NAME]
    return _db

def get_collection():
    """Get collection instance"""
    global _collection
    if _collection is None:
        db = get_database()
        _collection = db[settings.MONGODB_COLLECTION]
    return _collection

async def init_indexes():
    """Initialize database indexes"""
    try:
        collection = get_collection()
        
        # Create index on filename (unique)
        await collection.create_index("filename", unique=True)
        
        # Create index on user_id for faster queries
        await collection.create_index("user_id")
        
        # Create compound index
        await collection.create_index([("user_id", 1), ("filename", 1)])
        
        print("✅ MongoDB indexes created successfully")
        return True
    except Exception as e:
        print(f"⚠️  MongoDB index creation warning: {e}")
        return False

async def close_mongo_connection():
    """Close MongoDB connection"""
    global _client
    if _client:
        _client.close()
        _client = None
        print("✅ MongoDB connection closed")
