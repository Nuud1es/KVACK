"""bundle support: extractions and certificates keyed by (document, start page); job progress

Revision ID: 0004
Revises: 0003
"""

import sqlalchemy as sa
from alembic import op

revision = "0004"
down_revision = "0003"


def upgrade() -> None:
    for table in ("extractions", "calibration_certs"):
        op.add_column(
            table, sa.Column("start_page", sa.Integer(), server_default="1", nullable=False)
        )
        op.add_column(table, sa.Column("end_page", sa.Integer(), nullable=True))
        op.drop_constraint(f"{table}_pkey", table, type_="primary")
        op.create_primary_key(f"{table}_pkey", table, ["doc_id", "start_page"])
    op.create_table(
        "extraction_jobs",
        sa.Column("doc_id", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=12), nullable=False),
        sa.Column("total", sa.Integer(), nullable=False),
        sa.Column("done", sa.Integer(), nullable=False),
        sa.Column("failed", sa.Integer(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["doc_id"], ["documents.doc_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("doc_id"),
    )


def downgrade() -> None:
    op.drop_table("extraction_jobs")
    for table in ("calibration_certs", "extractions"):
        # keep only the first certificate of each document, so the old single-column key holds
        op.execute(
            f"DELETE FROM {table} t USING {table} u "
            "WHERE t.doc_id = u.doc_id AND t.start_page > u.start_page"
        )
        op.drop_constraint(f"{table}_pkey", table, type_="primary")
        op.create_primary_key(f"{table}_pkey", table, ["doc_id"])
        op.drop_column(table, "end_page")
        op.drop_column(table, "start_page")
