"""Fact Spec header and packaging rows, read in code from the page text

Revision ID: 0010
Revises: 0009
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0010"
down_revision = "0009"


def upgrade() -> None:
    op.create_table(
        "fact_specs",
        sa.Column(
            "doc_id",
            sa.String(64),
            sa.ForeignKey("documents.doc_id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("nav_code", sa.String(32), nullable=True),
        sa.Column("plu", sa.String(32), nullable=True),
        sa.Column("version", sa.Integer(), nullable=True),
        sa.Column("version_date", sa.Date(), nullable=True),
        sa.Column("brand", sa.String(128), nullable=True),
        sa.Column("site", sa.String(128), nullable=True),
        sa.Column("season", sa.String(64), nullable=True),
        sa.Column("product", sa.Text(), nullable=True),
        sa.Column("spec_type", sa.String(64), nullable=True),
        sa.Column("customer", sa.String(255), nullable=True),
        sa.Column("certification", sa.Text(), nullable=True),
        sa.Column("allergens", sa.Text(), nullable=True),
        sa.Column("status", sa.String(32), nullable=True),
    )
    op.create_index("ix_fact_specs_nav_code", "fact_specs", ["nav_code"])
    op.create_table(
        "spec_packaging",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "doc_id",
            sa.String(64),
            sa.ForeignKey("documents.doc_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("section", sa.String(12), nullable=False),
        sa.Column("code", sa.String(16), nullable=False),
        sa.Column("qty", sa.String(32), nullable=False, server_default=""),
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
        sa.Column("supplier", sa.String(255), nullable=False, server_default=""),
        sa.Column("dimension", sa.String(128), nullable=False, server_default=""),
        sa.Column("weight", sa.String(64), nullable=False, server_default=""),
        sa.Column("cells", postgresql.JSONB(), nullable=False),
        sa.Column("page", sa.Integer(), nullable=False, server_default="1"),
    )
    op.create_index("ix_spec_packaging_doc_id", "spec_packaging", ["doc_id"])
    op.create_index("ix_spec_packaging_code", "spec_packaging", ["code"])


def downgrade() -> None:
    op.drop_table("spec_packaging")
    op.drop_table("fact_specs")
