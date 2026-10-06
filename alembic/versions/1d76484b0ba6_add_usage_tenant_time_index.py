"""add usage tenant time index

Revision ID: 1d76484b0ba6
Revises: ec21542ad4e1
Create Date: 2026-10-05 21:59:12.300331

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "1d76484b0ba6"
down_revision: Union[str, Sequence[str], None] = "ec21542ad4e1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Plain CREATE INDEX blocks inserts on usage_logs for the whole build.
    # A failed concurrent build leaves an INVALID index: drop it before retrying.
    with op.get_context().autocommit_block():
        op.create_index(
            "ix_usage_logs_tenant_id_created_at",
            "usage_logs",
            ["tenant_id", "created_at"],
            postgresql_concurrently=True,
        )


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.drop_index(
            "ix_usage_logs_tenant_id_created_at",
            table_name="usage_logs",
            postgresql_concurrently=True,
        )
