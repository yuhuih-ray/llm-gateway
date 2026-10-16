"""Add optional tenant monthly budget.

Revision ID: 981c8c728b12
Revises: 0540092e10cf
"""

import sqlalchemy as sa

from alembic import op

revision = "981c8c728b12"
down_revision = "0540092e10cf"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "tenants", sa.Column("monthly_budget_usd", sa.Numeric(12, 2), nullable=True)
    )


def downgrade() -> None:
    op.drop_column("tenants", "monthly_budget_usd")
