"""Record exact response cache hits.

Revision ID: b7c913ef04aa
Revises: 981c8c728b12
"""

import sqlalchemy as sa

from alembic import op

revision = "b7c913ef04aa"
down_revision = "981c8c728b12"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # A constant default is a metadata-only change; alter the partitioned parent
    # so existing partitions and future partitions inherit the column.
    op.add_column(
        "usage_logs",
        sa.Column("cache_hit", sa.Boolean(), nullable=False, server_default=sa.false()),
    )


def downgrade() -> None:
    op.drop_column("usage_logs", "cache_hit")
