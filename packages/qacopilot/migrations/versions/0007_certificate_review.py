"""certificate review: flag unverified readings, record who confirmed or corrected them

Revision ID: 0007
Revises: 0006
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0007"
down_revision = "0006"


def upgrade() -> None:
    op.add_column(
        "calibration_certs",
        sa.Column("needs_review", sa.Boolean(), server_default=sa.false(), nullable=False),
    )
    op.add_column("calibration_certs", sa.Column("review_reason", sa.Text(), nullable=True))
    op.add_column("calibration_certs", sa.Column("verified_by", sa.String(128), nullable=True))
    op.add_column(
        "calibration_certs", sa.Column("verified_at", sa.DateTime(timezone=True), nullable=True)
    )
    op.create_index("ix_calibration_certs_needs_review", "calibration_certs", ["needs_review"])
    op.create_table(
        "certificate_reviews",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("doc_id", sa.String(64), nullable=False),
        sa.Column("start_page", sa.Integer(), nullable=False),
        sa.Column("reviewed_by", sa.String(128), nullable=False),
        sa.Column(
            "reviewed_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("action", sa.String(16), nullable=False),  # confirm | correct
        sa.Column("before", postgresql.JSONB(), nullable=False),
        sa.Column("after", postgresql.JSONB(), nullable=False),
        sa.Column("note", sa.Text(), nullable=True),
    )
    op.create_index("ix_certificate_reviews_cert", "certificate_reviews", ["doc_id", "start_page"])


def downgrade() -> None:
    op.drop_index("ix_certificate_reviews_cert", table_name="certificate_reviews")
    op.drop_table("certificate_reviews")
    op.drop_index("ix_calibration_certs_needs_review", table_name="calibration_certs")
    for col in ("verified_at", "verified_by", "review_reason", "needs_review"):
        op.drop_column("calibration_certs", col)
