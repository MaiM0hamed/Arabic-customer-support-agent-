"""PostgreSQL connection management via SQLAlchemy."""

from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from config import settings
from database.models import Base

engine = create_engine(settings.database_url, pool_pre_ping=True, future=True)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True, expire_on_commit=False)


def init_db() -> None:
    """Create all database tables defined on `Base.metadata`.

    Raises:
        sqlalchemy.exc.SQLAlchemyError: If the database connection fails.
    """
    Base.metadata.create_all(bind=engine)


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Yield a SQLAlchemy session, ensuring it is closed afterwards.

    Yields:
        Session: An active SQLAlchemy session.

    Raises:
        sqlalchemy.exc.SQLAlchemyError: If a database operation fails. The
            session is rolled back before the exception propagates.
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
