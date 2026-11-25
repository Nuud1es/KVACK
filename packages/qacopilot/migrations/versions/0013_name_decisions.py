"""name decisions: names a person confirmed are separate (not an alias of anything)

Revision ID: 0013
Revises: 0012
"""

import sqlalchemy as sa
from alembic import op

revision = "0013"
down_revision = "0012"


def upgrade() -> None:
    op.create_table(
        "name_decisions",
        sa.Column("kind", sa.String(12), primary_key=True),
        sa.Column("name_key", sa.String(255), primary_key=True),
        sa.Column("decision", sa.String(16), nullable=False),
        sa.Column("decided_by", sa.String(128), nullable=True),
        sa.Column(
            "decided_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )


def downgrade() -> None:
    op.drop_table("name_decisions")
