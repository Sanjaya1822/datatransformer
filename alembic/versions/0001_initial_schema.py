"""Initial database schema for Candidate Data Transformer.

Revision ID: 0001_initial
Revises: 
Create Date: 2026-10-28 20:40:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from data_transformer.db.database import Base


# revision identifiers, used by Alembic.
revision: str = "0001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Use metadata create_all for full database dialect compatibility
    bind = op.get_bind()
    Base.metadata.create_all(bind=bind)


def downgrade() -> None:
    bind = op.get_bind()
    Base.metadata.drop_all(bind=bind)
