from sqlalchemy import or_, BinaryExpression
from sqlalchemy.orm import sessionmaker
from typing import Generic, TypeVar, Type, List, Optional, Iterable
from contextlib import contextmanager
from models.base import Base

# Step 2: Define a generic type var bound to Base
T = TypeVar('T', bound=Base)

# Step 3: Generic BaseDao
class BaseDao(Generic[T]):
    def __init__(self, session_factory: sessionmaker, model: Type[T]):
        self.session_factory = session_factory
        self.model = model  # The SQLAlchemy model class, e.g., Auth
    
    @contextmanager
    def session_scope(self):
        """Provide a transactional scope around a series of operations."""
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def add(self, obj: T) -> int:
        """Add a single object to the database and return only the ID.
        
        This follows the pattern of returning just the ID and letting the caller
        load a fresh instance if needed.
        """
        with self.session_scope() as session:
            session.add(obj)
            session.flush()  # Flush to get the ID assigned
            # Return just the ID
            return obj.id
        
    def add_return_obj(self, obj: T) -> T:
        """Add a single object to the database and return the full object.
        
        Warning: The returned object will be detached from its session.
        Use add() and get_by_id() instead for most use cases.
        """
        with self.session_scope() as session:
            session.add(obj)
            session.flush()  # Flush to get the ID assigned
            # Return the fully populated object
            return obj
        
    def add_all(self, objs: Iterable[T]) -> List[int]:
        """Add multiple objects to the database and return their IDs."""
        with self.session_scope() as session:
            session.add_all(objs)
            session.flush()  # Flush to get IDs assigned
            # Return just the IDs
            return [obj.id for obj in objs]
    
    def add_all_return_obj(self, objs: Iterable[T]) -> List[T]:
        """Add multiple objects to the database and return all the objects.
        
        Warning: The returned objects will be detached from their session.
        Use add_all() and get_by_ids() instead for most use cases.
        """
        with self.session_scope() as session:
            session.add_all(objs)
            session.flush()  # Flush to get IDs assigned
            # Return all the objects
            return list(objs)

    def get_all(self) -> Iterable[T]:
        with self.session_scope() as session:
            return session.query(self.model).all()

    def get_by_id(self, id: int) -> Optional[T]:
        with self.session_scope() as session:
            return session.query(self.model).filter(self.model.id == id).first()
    
    def get_by_ids(self, ids: List[int]) -> List[T]:
        with self.session_scope() as session:
            return session.query(self.model).filter(self.model.id.in_(ids)).all()


    def delete_by_ids(self, ids: List[int]) -> None:
        """Delete objects by their IDs, respecting relationship cascades."""
        with self.session_scope() as session:
            # Fetch the objects first to properly trigger cascade behavior
            objects = session.query(self.model).filter(self.model.id.in_(ids)).all()
            for obj in objects:
                session.delete(obj)  # This will trigger cascade behavior

    def update_by_ids(self, ids: List[int], values: dict) -> None:
        with self.session_scope() as session:
            session.query(self.model).filter(self.model.id.in_(ids)).update(values)

    def update_by_id(self, id: int, values: dict) -> bool:
        with self.session_scope() as session:
            result = session.query(self.model).filter(self.model.id == id).update(values)
            return result > 0
        
    # Conditions are concatenated with AND
    def get_all_by_and_conditions(self, conditions: List[BinaryExpression], offset: int = 0, limit: Optional[int] = None) -> List[T]:
        with self.session_scope() as session:
            query = session.query(self.model)
            if conditions:
                query = query.filter(*conditions)
            query = query.offset(offset)
            if limit is not None:
                query = query.limit(limit)
            return query.all()
        
    # Conditions are concatenated with AND
    def get_first_by_and_conditions(self, conditions: List[BinaryExpression]) -> Optional[T]:
        with self.session_scope() as session:
            query = session.query(self.model)
            if conditions:
                query = query.filter(*conditions)
            return query.first()
    
    # Conditions are concatenated with OR
    def get_all_by_or_conditions(self, conditions: List[BinaryExpression], offset: int = 0, limit: Optional[int] = None) -> List[T]:
        with self.session_scope() as session:
            query = session.query(self.model)
            if conditions:
                query = query.filter(or_(*conditions))
            query = query.offset(offset)
            if limit is not None:
                query = query.limit(limit)
            return query.all()
    
    # Conditions are concatenated with OR
    def get_first_by_or_conditions(self, conditions: List[BinaryExpression]) -> Optional[T]:
        with self.session_scope() as session:
            query = session.query(self.model)
            if conditions:
                query = query.filter(or_(*conditions))
            return query.first()

    def count_by_and_conditions(self, conditions: List[BinaryExpression]) -> int:
        """Count records matching all conditions (concatenated with AND).
        
        Args:
            conditions: List of SQLAlchemy binary expressions to filter by
            
        Returns:
            int: Number of records matching all conditions
        """
        with self.session_scope() as session:
            query = session.query(self.model)
            if conditions:
                query = query.filter(*conditions)
            return query.count()