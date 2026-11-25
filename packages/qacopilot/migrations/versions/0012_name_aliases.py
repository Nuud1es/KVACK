"""name aliases: spellings that are one customer, supplier or person

Revision ID: 0012
Revises: 0011
"""

import sqlalchemy as sa
from alembic import op

revision = "0012"
down_revision = "0011"


def upgrade() -> None:
    op.create_table(
        "name_aliases",
        sa.Column("kind", sa.String(12), primary_key=True),
        sa.Column("raw_key", sa.String(255), primary_key=True),
        sa.Column("canonical", sa.String(255), nullable=False),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("created_by", sa.String(128), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )


def downgrade() -> None:
    op.drop_table("name_aliases")
