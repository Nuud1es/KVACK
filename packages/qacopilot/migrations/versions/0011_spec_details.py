"""what a Fact Spec says: storage, shelf life, pallet, metal detection, tare, steps, ingredients,
and the version history (who changed what)

Revision ID: 0011
Revises: 0010
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0011"
down_revision = "0010"

_COLUMNS = [
    ("weight", sa.String(64)),
    ("weight_type", sa.String(32)),
    ("storage", sa.Text()),
    ("temp_min_c", sa.Float()),
    ("temp_max_c", sa.Float()),
    ("max_life_text", sa.Text()),
    ("max_life_days", sa.Integer()),
    ("life_basis", sa.String(32)),
    ("life_notes", sa.Text()),
    ("min_life_text", sa.Text()),
    ("min_life_days", sa.Integer()),
    ("boxes_per_layer", sa.Integer()),
    ("layers_per_pallet", sa.Integer()),
    ("cases_per_pallet", sa.Integer()),
    ("pallet_type", sa.String(64)),
    ("pallet", postgresql.JSONB()),
    ("metal_detection", postgresql.JSONB()),
    ("metal_detection_text", sa.Text()),
    ("ferrous_mm", sa.Float()),
    ("non_ferrous_mm", sa.Float()),
    ("stainless_mm", sa.Float()),
    ("pack_tare_g", sa.Float()),
    ("case_tare_g", sa.Float()),
    ("product_weight", sa.String(64)),
    ("process_steps", postgresql.JSONB()),
    ("raw_materials", postgresql.JSONB()),
]


def upgrade() -> None:
    for name, kind in _COLUMNS:
        op.add_column("fact_specs", sa.Column(name, kind, nullable=True))
    op.create_table(
        "spec_versions",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "doc_id",
            sa.String(64),
            sa.ForeignKey("documents.doc_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("version", sa.Integer(), nullable=True),
        sa.Column("person", sa.String(128), nullable=False),
        sa.Column("pages", sa.String(64), nullable=False, server_default=""),
        sa.Column("amends", sa.Text(), nullable=False, server_default=""),
        sa.Column("changed_on", sa.Date(), nullable=True),
    )
    op.create_index("ix_spec_versions_doc_id", "spec_versions", ["doc_id"])
    op.create_index("ix_spec_versions_person", "spec_versions", ["person"])


def downgrade() -> None:
    op.drop_table("spec_versions")
    for name, _ in reversed(_COLUMNS):
        op.drop_column("fact_specs", name)
