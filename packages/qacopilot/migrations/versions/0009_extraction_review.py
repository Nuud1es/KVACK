"""review flags on extractions, so CoAs and specs can be confirmed like certificates

Revision ID: 0009
Revises: 0008
"""

import sqlalchemy as sa
from alembic import op

revision = "0009"
down_revision = "0008"


def upgrade() -> None:
    op.add_column(
        "extractions",
        sa.Column("needs_review", sa.Boolean(), server_default=sa.false(), nullable=False),
    )
    op.add_column("extractions", sa.Column("review_reason", sa.Text(), nullable=True))
    op.add_column("extractions", sa.Column("verified_by", sa.String(128), nullable=True))
    op.add_column(
        "extractions", sa.Column("verified_at", sa.DateTime(timezone=True), nullable=True)
    )
    op.create_index("ix_extractions_needs_review", "extractions", ["needs_review"])


def downgrade() -> None:
    op.drop_index("ix_extractions_needs_review", table_name="extractions")
    for column in ("verified_at", "verified_by", "review_reason", "needs_review"):
        op.drop_column("extractions", column)
