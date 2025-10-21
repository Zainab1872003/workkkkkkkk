# # core/database.py
# """
# MongoDB database connection and initialization
# """
# from motor.motor_asyncio import AsyncIOMotorClient
# from core.config import settings

# # MongoDB client
# _client = None
# _db = None
# _collection = None

# def get_mongo_client():
#     """Get MongoDB client instance"""
#     global _client
#     if _client is None:
#         _client = AsyncIOMotorClient(settings.MONGODB_URL)
#     return _client

# def get_database():
#     """Get database instance"""
#     global _db
#     if _db is None:
#         client = get_mongo_client()
#         _db = client[settings.DATABASE_NAME]
#     return _db

# def get_collection():
#     """Get collection instance"""
#     global _collection
#     if _collection is None:
#         db = get_database()
#         _collection = db[settings.MONGODB_COLLECTION]
#     return _collection

# async def init_indexes():
#     """Initialize database indexes"""
#     try:
#         collection = get_collection()
        
#         # Create index on filename (unique)
#         await collection.create_index("filename", unique=True)
        
#         # Create index on user_id for faster queries
#         await collection.create_index("user_id")
        
#         # Create compound index
#         await collection.create_index([("user_id", 1), ("filename", 1)])
        
#         print("✅ MongoDB indexes created successfully")
#         return True
#     except Exception as e:
#         print(f"⚠️  MongoDB index creation warning: {e}")
#         return False

# async def close_mongo_connection():
#     """Close MongoDB connection"""
#     global _client
#     if _client:
#         _client.close()
#         _client = None
#         print("✅ MongoDB connection closed")


# core/database.py
"""
MongoDB database connection and collection management
"""
import logging
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import CollectionInvalid

from core.config import settings

logger = logging.getLogger(__name__)

# Global MongoDB client
mongo_client: AsyncIOMotorClient = None


async def connect_mongo():
    """Connect to MongoDB"""
    global mongo_client
    try:
        mongo_client = AsyncIOMotorClient(settings.MONGODB_URL)
        # Test connection
        await mongo_client.admin.command('ping')
        logger.info(f"✅ Connected to MongoDB: {settings.DATABASE_NAME}")
    except Exception as e:
        logger.error(f"❌ MongoDB connection failed: {e}")
        raise


async def close_mongo_connection():
    """Close MongoDB connection"""
    global mongo_client
    if mongo_client:
        mongo_client.close()
        logger.info("✅ Closed MongoDB connection")


async def init_indexes():
    """Initialize MongoDB indexes for all collections"""
    global mongo_client
    
    if mongo_client is None:
        await connect_mongo()
    
    db = mongo_client[settings.DATABASE_NAME]
    
    try:
        # Create indexes for documents collection
        documents_col = db["documents"]
        await documents_col.create_index([("filename", 1), ("user_id", 1)], unique=True)
        await documents_col.create_index([("user_id", 1)])
        await documents_col.create_index([("upload_date", -1)])
        logger.info("✅ Created indexes for 'documents' collection")
        
        # Create indexes for usecases collection
        usecases_col = db["usecases"]
        await usecases_col.create_index([("id", 1)], unique=True)
        await usecases_col.create_index([("category", 1)])
        await usecases_col.create_index([("is_active", 1)])
        await usecases_col.create_index([("created_at", -1)])
        logger.info("✅ Created indexes for 'usecases' collection")
        
    except Exception as e:
        logger.warning(f"⚠️ Index creation warning: {e}")


def get_collection(collection_name: str = "documents"):
    """
    Get MongoDB collection by name.
    Automatically creates the collection if it doesn't exist.
    
    Args:
        collection_name: Name of the collection (default: "documents")
        
    Returns:
        AsyncIOMotorCollection
    """
    if mongo_client is None:
        raise Exception("MongoDB not initialized. Call init_indexes() first.")
    
    db = mongo_client[settings.DATABASE_NAME]
    
    # MongoDB automatically creates collections when you first write to them
    # No need to explicitly create, just return the collection reference
    collection = db[collection_name]
    
    logger.debug(f"📦 Accessing collection: {collection_name}")
    
    return collection


async def collection_exists(collection_name: str) -> bool:
    """
    Check if a collection exists in MongoDB.
    
    Args:
        collection_name: Name of the collection
        
    Returns:
        bool: True if collection exists, False otherwise
    """
    if mongo_client is None:
        raise Exception("MongoDB not initialized")
    
    db = mongo_client[settings.DATABASE_NAME]
    collections = await db.list_collection_names()
    
    return collection_name in collections


async def create_collection_if_not_exists(collection_name: str, indexes: list = None):
    """
    Create a MongoDB collection if it doesn't exist and optionally add indexes.
    
    Args:
        collection_name: Name of the collection to create
        indexes: Optional list of index specifications
                 Example: [{"keys": [("field_name", 1)], "unique": True}]
    """
    if mongo_client is None:
        raise Exception("MongoDB not initialized")
    
    db = mongo_client[settings.DATABASE_NAME]
    
    # Check if collection exists
    exists = await collection_exists(collection_name)
    
    if exists:
        logger.info(f"📦 Collection '{collection_name}' already exists")
        return db[collection_name]
    
    # Create collection explicitly
    try:
        collection = await db.create_collection(collection_name)
        logger.info(f"✅ Created collection: {collection_name}")
        
        # Create indexes if provided
        if indexes:
            for index_spec in indexes:
                keys = index_spec.get("keys")
                unique = index_spec.get("unique", False)
                name = index_spec.get("name")
                
                await collection.create_index(keys, unique=unique, name=name)
                logger.info(f"✅ Created index on {collection_name}: {keys}")
        
        return collection
        
    except CollectionInvalid:
        # Collection was created between check and create (race condition)
        logger.info(f"📦 Collection '{collection_name}' already exists (race condition)")
        return db[collection_name]
    except Exception as e:
        logger.error(f"❌ Failed to create collection '{collection_name}': {e}")
        raise


async def get_or_create_collection(collection_name: str, indexes: list = None):
    """
    Get a collection, creating it with indexes if it doesn't exist.
    
    This is a convenience function that combines get_collection and create_collection_if_not_exists.
    
    Args:
        collection_name: Name of the collection
        indexes: Optional list of index specifications
        
    Returns:
        AsyncIOMotorCollection
    """
    if mongo_client is None:
        raise Exception("MongoDB not initialized")
    
    # Check if exists
    exists = await collection_exists(collection_name)
    
    if not exists:
        # Create with indexes
        await create_collection_if_not_exists(collection_name, indexes)
        logger.info(f"✅ Collection '{collection_name}' created")
    else:
        logger.debug(f"📦 Using existing collection: {collection_name}")
    
    return get_collection(collection_name)
